"""
System Tools Adapter

Provides basic system operations that the unified agent needs:
- File reading/writing/viewing
- Command execution
- Directory listing

These are fundamental capabilities that cannot be created by the self-evolving
agent due to security restrictions in the tool registry.
"""

import os
import subprocess
import shlex
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import CookbookAdapter, AdapterAction


class SystemToolsAdapter(CookbookAdapter):
    """
    Adapter providing basic system tools for file operations and command execution.
    
    Security Notes:
    - File operations are restricted to a configurable working directory
    - Command execution has configurable timeout and can be disabled
    - No shell expansion by default (uses shlex for argument parsing)
    """
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        allow_commands: bool = True,
        command_timeout: int = 60,
        max_file_size: int = 1024 * 1024,  # 1MB
        max_output_size: int = 100000,  # 100KB
    ):
        super().__init__()
        self._working_dir = working_dir
        self._allow_commands = allow_commands
        self._command_timeout = command_timeout
        self._max_file_size = max_file_size
        self._max_output_size = max_output_size
    
    @property
    def name(self) -> str:
        return "system-tools"
    
    @property
    def description(self) -> str:
        return "Basic system operations: read/write files, run commands, list directories"
    
    @property
    def tags(self) -> List[str]:
        return ["system", "file", "command", "core"]
    
    @property
    def dependencies(self) -> List[str]:
        return []
    
    @property
    def actions(self) -> List[AdapterAction]:
        actions = [
            AdapterAction(
                name="read_file",
                description="Read the contents of a file",
                parameters={
                    "path": {
                        "type": "str",
                        "description": "Path to the file to read",
                        "required": True
                    },
                    "start_line": {
                        "type": "int",
                        "description": "Starting line number (1-indexed, optional)",
                        "required": False,
                        "default": None
                    },
                    "end_line": {
                        "type": "int",
                        "description": "Ending line number (inclusive, optional)",
                        "required": False,
                        "default": None
                    }
                }
            ),
            AdapterAction(
                name="write_file",
                description="Write content to a file (creates parent directories if needed)",
                parameters={
                    "path": {
                        "type": "str",
                        "description": "Path to the file to write",
                        "required": True
                    },
                    "content": {
                        "type": "str",
                        "description": "Content to write to the file",
                        "required": True
                    },
                    "append": {
                        "type": "bool",
                        "description": "If true, append to file instead of overwriting",
                        "required": False,
                        "default": False
                    }
                }
            ),
            AdapterAction(
                name="list_directory",
                description="List files and directories in a path",
                parameters={
                    "path": {
                        "type": "str",
                        "description": "Directory path to list (default: current directory)",
                        "required": False,
                        "default": "."
                    },
                    "pattern": {
                        "type": "str",
                        "description": "Glob pattern to filter files (e.g., '*.py')",
                        "required": False,
                        "default": "*"
                    },
                    "recursive": {
                        "type": "bool",
                        "description": "If true, list recursively",
                        "required": False,
                        "default": False
                    }
                }
            ),
            AdapterAction(
                name="file_exists",
                description="Check if a file or directory exists",
                parameters={
                    "path": {
                        "type": "str",
                        "description": "Path to check",
                        "required": True
                    }
                }
            ),
            AdapterAction(
                name="delete_file",
                description="Delete a file",
                parameters={
                    "path": {
                        "type": "str",
                        "description": "Path to the file to delete",
                        "required": True
                    }
                }
            ),
            AdapterAction(
                name="create_directory",
                description="Create a directory (including parent directories)",
                parameters={
                    "path": {
                        "type": "str",
                        "description": "Directory path to create",
                        "required": True
                    }
                }
            ),
        ]
        
        if self._allow_commands:
            actions.append(
                AdapterAction(
                    name="run_command",
                    description="Execute a shell command and return its output",
                    parameters={
                        "command": {
                            "type": "str",
                            "description": "The command to execute",
                            "required": True
                        },
                        "working_dir": {
                            "type": "str",
                            "description": "Working directory for the command",
                            "required": False,
                            "default": None
                        },
                        "timeout": {
                            "type": "int",
                            "description": "Timeout in seconds",
                            "required": False,
                            "default": None
                        }
                    }
                )
            )
        
        return actions
    
    def initialize(self, shared: Dict[str, Any]) -> None:
        """Initialize with configuration from shared state."""
        if self._working_dir is None:
            self._working_dir = shared.get("working_dir", os.getcwd())
        
        # Allow overriding settings from shared
        if "allow_commands" in shared:
            self._allow_commands = shared["allow_commands"]
        if "command_timeout" in shared:
            self._command_timeout = shared["command_timeout"]
        
        self._initialized = True
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the working directory."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self._working_dir) / p
        return p.resolve()
    
    def _truncate_output(self, output: str, max_size: Optional[int] = None) -> str:
        """Truncate output if it exceeds max size."""
        max_size = max_size or self._max_output_size
        if len(output) > max_size:
            truncated = output[:max_size]
            return truncated + f"\n\n... [truncated, {len(output) - max_size} bytes omitted]"
        return output
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the requested action."""
        try:
            if action_name == "read_file":
                return self._read_file(params)
            elif action_name == "write_file":
                return self._write_file(params)
            elif action_name == "list_directory":
                return self._list_directory(params)
            elif action_name == "file_exists":
                return self._file_exists(params)
            elif action_name == "delete_file":
                return self._delete_file(params)
            elif action_name == "create_directory":
                return self._create_directory(params)
            elif action_name == "run_command":
                if not self._allow_commands:
                    return {"success": False, "error": "Command execution is disabled"}
                return self._run_command(params)
            else:
                return {"success": False, "error": f"Unknown action: {action_name}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}
    
    def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read file contents."""
        path = self._resolve_path(params["path"])
        start_line = params.get("start_line")
        end_line = params.get("end_line")
        
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        if not path.is_file():
            return {"success": False, "error": f"Not a file: {path}"}
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self._max_file_size:
            return {
                "success": False,
                "error": f"File too large ({file_size} bytes > {self._max_file_size} bytes max)"
            }
        
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except UnicodeDecodeError:
            # Try reading as binary for non-text files
            return {
                "success": False,
                "error": "File appears to be binary, cannot read as text"
            }
        
        # Handle line range
        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)
            
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else total_lines
            
            start_idx = max(0, min(start_idx, total_lines))
            end_idx = max(0, min(end_idx, total_lines))
            
            content = "".join(lines[start_idx:end_idx])
            line_info = f" (lines {start_idx + 1}-{end_idx} of {total_lines})"
        else:
            line_info = f" ({len(content.splitlines())} lines)"
        
        content = self._truncate_output(content)
        
        return {
            "success": True,
            "result": content,
            "context_update": f"Read file: {path}{line_info}"
        }
    
    def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a file."""
        path = self._resolve_path(params["path"])
        content = params["content"]
        append = params.get("append", False)
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        
        action = "Appended to" if append else "Wrote"
        return {
            "success": True,
            "result": f"Successfully wrote {len(content)} bytes to {path}",
            "context_update": f"{action} file: {path} ({len(content)} bytes)"
        }
    
    def _list_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents."""
        path = self._resolve_path(params.get("path", "."))
        pattern = params.get("pattern", "*")
        recursive = params.get("recursive", False)
        
        if not path.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        
        if not path.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}
        
        try:
            if recursive:
                items = list(path.rglob(pattern))
            else:
                items = list(path.glob(pattern))
            
            # Sort and format
            items = sorted(items)
            
            entries = []
            for item in items[:500]:  # Limit results
                rel_path = item.relative_to(path) if item.is_relative_to(path) else item
                entry_type = "dir" if item.is_dir() else "file"
                size = item.stat().st_size if item.is_file() else 0
                entries.append({
                    "name": str(rel_path),
                    "type": entry_type,
                    "size": size
                })
            
            # Format for context
            lines = []
            for e in entries[:50]:
                prefix = "📁" if e["type"] == "dir" else "📄"
                size_str = f" ({e['size']} bytes)" if e["type"] == "file" else ""
                lines.append(f"{prefix} {e['name']}{size_str}")
            
            context = f"Directory listing of {path}:\n" + "\n".join(lines)
            if len(entries) > 50:
                context += f"\n... and {len(entries) - 50} more items"
            
            return {
                "success": True,
                "result": entries,
                "context_update": context
            }
            
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path}"}
    
    def _file_exists(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if file exists."""
        path = self._resolve_path(params["path"])
        exists = path.exists()
        
        if exists:
            if path.is_file():
                result = f"File exists: {path}"
            elif path.is_dir():
                result = f"Directory exists: {path}"
            else:
                result = f"Path exists (other type): {path}"
        else:
            result = f"Does not exist: {path}"
        
        return {
            "success": True,
            "result": exists,
            "context_update": result
        }
    
    def _delete_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a file."""
        path = self._resolve_path(params["path"])
        
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        if not path.is_file():
            return {"success": False, "error": f"Not a file (use rm -r for directories): {path}"}
        
        path.unlink()
        
        return {
            "success": True,
            "result": f"Deleted: {path}",
            "context_update": f"Deleted file: {path}"
        }
    
    def _create_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a directory."""
        path = self._resolve_path(params["path"])
        
        if path.exists():
            if path.is_dir():
                return {
                    "success": True,
                    "result": f"Directory already exists: {path}",
                    "context_update": f"Directory exists: {path}"
                }
            else:
                return {"success": False, "error": f"Path exists but is not a directory: {path}"}
        
        path.mkdir(parents=True, exist_ok=True)
        
        return {
            "success": True,
            "result": f"Created directory: {path}",
            "context_update": f"Created directory: {path}"
        }
    
    def _run_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command."""
        command = params["command"]
        working_dir = params.get("working_dir")
        timeout = params.get("timeout") or self._command_timeout
        
        if working_dir:
            cwd = self._resolve_path(working_dir)
        else:
            cwd = Path(self._working_dir)
        
        if not cwd.exists():
            return {"success": False, "error": f"Working directory not found: {cwd}"}
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd)
            )
            
            stdout = self._truncate_output(result.stdout) if result.stdout else ""
            stderr = self._truncate_output(result.stderr) if result.stderr else ""
            
            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            
            output = "\n\n".join(output_parts) if output_parts else "(no output)"
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "result": {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode
                    },
                    "context_update": f"Command `{command}` (exit {result.returncode}):\n{output}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Command failed with exit code {result.returncode}",
                    "result": {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode
                    },
                    "context_update": f"Command `{command}` failed (exit {result.returncode}):\n{output}"
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout} seconds"}
        except FileNotFoundError as e:
            return {"success": False, "error": f"Command not found: {e}"}


def get_system_tools_adapter(
    working_dir: Optional[str] = None,
    allow_commands: bool = True
) -> SystemToolsAdapter:
    """Create a system tools adapter with the specified configuration."""
    return SystemToolsAdapter(
        working_dir=working_dir,
        allow_commands=allow_commands
    )
