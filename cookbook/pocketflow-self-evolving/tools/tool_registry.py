"""
Tool Registry - Persistent storage and retrieval for dynamically created tools.

Supports:
- Tool registration with metadata, source code, and versioning
- Semantic search via embeddings
- Safe execution with sandboxing
- Tool validation and testing
"""

import json
import hashlib
import time
import os
import threading
import fcntl
import sqlite3
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
import traceback


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {type, description, required, default}
    return_type: str
    return_description: str
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)  # [{input: ..., output: ...}]
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    source_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolMetadata":
        return cls(**data)


@dataclass
class RegisteredTool:
    """A complete registered tool with metadata and source code."""
    metadata: ToolMetadata
    source_code: str
    embedding: Optional[List[float]] = None
    is_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "source_code": self.source_code,
            "embedding": self.embedding,
            "is_enabled": self.is_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredTool":
        return cls(
            metadata=ToolMetadata.from_dict(data["metadata"]),
            source_code=data["source_code"],
            embedding=data.get("embedding"),
            is_enabled=data.get("is_enabled", True)
        )


class ToolValidationError(Exception):
    """Raised when tool validation fails."""
    pass


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found."""
    pass


class ToolRegistry:
    """
    Thread-safe, persistent registry for dynamically created tools.
    
    Supports SQLite backend with file locking for concurrent access,
    embedding-based semantic search, and safe sandboxed execution.
    """
    
    _instances: Dict[str, "ToolRegistry"] = {}
    _instance_lock = threading.Lock()
    
    @classmethod
    def _clear_instances(cls):
        """Clear singleton instances - for testing only."""
        with cls._instance_lock:
            cls._instances.clear()
    
    # Singleton per database path
    def __new__(cls, db_path: str = "tool_registry.db", *args, **kwargs):
        abs_path = os.path.abspath(db_path)
        with cls._instance_lock:
            if abs_path not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[abs_path] = instance
            return cls._instances[abs_path]
    
    def __init__(
        self, 
        db_path: str = "tool_registry.db",
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        max_execution_time: float = 30.0,
        allowed_imports: Optional[List[str]] = None
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.db_path = os.path.abspath(db_path)
        self.lock_path = self.db_path + ".lock"
        self.embedding_fn = embedding_fn
        self.max_execution_time = max_execution_time
        self.allowed_imports = allowed_imports or [
            "math", "re", "json", "datetime", "collections", 
            "itertools", "functools", "operator", "string",
            "hashlib", "base64", "urllib.parse", "statistics"
        ]
        
        self._local = threading.local()
        self._write_lock = threading.Lock()
        
        self._init_database()
        self._initialized = True
    
    @contextmanager
    def _file_lock(self, exclusive: bool = False):
        """Acquire file lock for database operations."""
        lock_fd = None
        try:
            lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT)
            if exclusive:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            else:
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
            yield
        finally:
            if lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_database(self):
        """Initialize database schema."""
        with self._file_lock(exclusive=True):
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    name TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    source_code TEXT NOT NULL,
                    embedding_json TEXT,
                    is_enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_code TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(tool_name, version)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    error TEXT,
                    execution_time_ms INTEGER,
                    success INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_enabled ON tools(is_enabled)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_name ON tool_versions(tool_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_name ON execution_logs(tool_name)")
            
            conn.commit()
    
    def _compute_source_hash(self, source_code: str) -> str:
        """Compute hash of source code for change detection."""
        normalized = source_code.strip().replace("\r\n", "\n")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    
    def _validate_tool_source(self, source_code: str, metadata: ToolMetadata) -> Tuple[bool, str]:
        """
        Validate tool source code for safety and correctness.
        Returns (is_valid, error_message).
        """
        if not source_code or not source_code.strip():
            return False, "Source code cannot be empty"
        
        function_name = metadata.name
        if f"def {function_name}(" not in source_code:
            return False, f"Source code must define function '{function_name}'"
        
        # Security patterns - must be standalone calls, not part of identifiers
        # Use regex to avoid false positives like "concurrent_exec" matching "exec("
        import re
        dangerous_patterns = [
            (r'\bexec\s*\(', "Dynamic code execution not allowed"),
            (r'\beval\s*\(', "Dynamic evaluation not allowed"),
            (r'__import__', "Dynamic imports not allowed"),
            (r'(?<![_a-zA-Z])open\s*\(', "File operations not allowed - use approved utilities"),
            (r'\bos\.system\b', "System calls not allowed"),
            (r'\bsubprocess\b', "Subprocess not allowed"),
            (r'\bcompile\s*\(', "Code compilation not allowed"),
        ]
        
        for pattern, reason in dangerous_patterns:
            if re.search(pattern, source_code):
                return False, f"Security violation: {reason}"
        
        import_lines = [l for l in source_code.split("\n") if l.strip().startswith("import ") or l.strip().startswith("from ")]
        for line in import_lines:
            line_clean = line.strip()
            if line_clean.startswith("import "):
                module = line_clean.split()[1].split(".")[0]
            elif line_clean.startswith("from "):
                module = line_clean.split()[1].split(".")[0]
            else:
                continue
            
            if module not in self.allowed_imports:
                return False, f"Import of '{module}' not in allowed list: {self.allowed_imports}"
        
        try:
            compile(source_code, "<tool>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        return True, ""
    
    def _compute_embedding(self, tool: RegisteredTool) -> Optional[List[float]]:
        """Compute embedding for semantic search."""
        if self.embedding_fn is None:
            return None
        
        try:
            text_for_embedding = f"""
Tool: {tool.metadata.name}
Description: {tool.metadata.description}
Parameters: {json.dumps(tool.metadata.parameters)}
Tags: {', '.join(tool.metadata.tags)}
Examples: {json.dumps(tool.metadata.examples[:3]) if tool.metadata.examples else 'None'}
"""
            return self.embedding_fn(text_for_embedding.strip())
        except Exception as e:
            print(f"Warning: Failed to compute embedding for {tool.metadata.name}: {e}")
            return None
    
    def register(
        self,
        name: str,
        description: str,
        source_code: str,
        parameters: Dict[str, Dict[str, Any]],
        return_type: str = "Any",
        return_description: str = "",
        tags: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        validate: bool = True,
        test_examples: bool = True
    ) -> RegisteredTool:
        """
        Register a new tool or update existing one.
        
        Args:
            name: Unique tool name (also the function name)
            description: Human-readable description
            source_code: Python source code defining the function
            parameters: Parameter specifications
            return_type: Return type annotation
            return_description: Description of return value
            tags: Searchable tags
            examples: Example inputs/outputs for testing
            validate: Whether to validate source code
            test_examples: Whether to run examples as tests
            
        Returns:
            The registered tool
            
        Raises:
            ToolValidationError: If validation fails
        """
        tags = tags or []
        examples = examples or []
        
        metadata = ToolMetadata(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
            return_description=return_description,
            tags=tags,
            examples=examples,
            source_hash=self._compute_source_hash(source_code)
        )
        
        if validate:
            is_valid, error = self._validate_tool_source(source_code, metadata)
            if not is_valid:
                raise ToolValidationError(f"Tool validation failed: {error}")
        
        tool = RegisteredTool(metadata=metadata, source_code=source_code)
        
        if test_examples and examples:
            for i, example in enumerate(examples):
                try:
                    result = self._execute_tool_internal(tool, example.get("input", {}))
                    expected = example.get("output")
                    if expected is not None and result != expected:
                        raise ToolValidationError(
                            f"Example {i+1} failed: expected {expected}, got {result}"
                        )
                except ToolExecutionError as e:
                    raise ToolValidationError(f"Example {i+1} execution failed: {e}")
        
        tool.embedding = self._compute_embedding(tool)
        
        with self._write_lock:
            with self._file_lock(exclusive=True):
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT metadata_json FROM tools WHERE name = ?", (name,))
                existing = cursor.fetchone()
                
                if existing:
                    old_metadata = json.loads(existing["metadata_json"])
                    metadata.version = old_metadata.get("version", 1) + 1
                    metadata.created_at = old_metadata.get("created_at", metadata.created_at)
                    metadata.usage_count = old_metadata.get("usage_count", 0)
                    metadata.success_count = old_metadata.get("success_count", 0)
                    metadata.failure_count = old_metadata.get("failure_count", 0)
                    
                    cursor.execute("""
                        INSERT INTO tool_versions (tool_name, version, metadata_json, source_code, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        name,
                        old_metadata.get("version", 1),
                        existing["metadata_json"],
                        cursor.execute("SELECT source_code FROM tools WHERE name = ?", (name,)).fetchone()["source_code"],
                        datetime.utcnow().isoformat()
                    ))
                
                now = datetime.utcnow().isoformat()
                cursor.execute("""
                    INSERT OR REPLACE INTO tools (name, metadata_json, source_code, embedding_json, is_enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    name,
                    json.dumps(metadata.to_dict()),
                    source_code,
                    json.dumps(tool.embedding) if tool.embedding else None,
                    1,
                    metadata.created_at,
                    now
                ))
                
                conn.commit()
        
        return tool
    
    def get(self, name: str, version: Optional[int] = None) -> RegisteredTool:
        """
        Get a tool by name, optionally at a specific version.
        
        Args:
            name: Tool name
            version: Specific version (None for latest)
            
        Returns:
            The requested tool
            
        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        with self._file_lock(exclusive=False):
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if version is None:
                cursor.execute("""
                    SELECT metadata_json, source_code, embedding_json, is_enabled
                    FROM tools WHERE name = ?
                """, (name,))
                row = cursor.fetchone()
                
                if not row:
                    raise ToolNotFoundError(f"Tool '{name}' not found")
                
                return RegisteredTool(
                    metadata=ToolMetadata.from_dict(json.loads(row["metadata_json"])),
                    source_code=row["source_code"],
                    embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
                    is_enabled=bool(row["is_enabled"])
                )
            else:
                cursor.execute("""
                    SELECT metadata_json, source_code
                    FROM tool_versions WHERE tool_name = ? AND version = ?
                """, (name, version))
                row = cursor.fetchone()
                
                if not row:
                    raise ToolNotFoundError(f"Tool '{name}' version {version} not found")
                
                return RegisteredTool(
                    metadata=ToolMetadata.from_dict(json.loads(row["metadata_json"])),
                    source_code=row["source_code"],
                    embedding=None,
                    is_enabled=True
                )
    
    def exists(self, name: str) -> bool:
        """Check if a tool exists."""
        try:
            self.get(name)
            return True
        except ToolNotFoundError:
            return False
    
    def list_tools(
        self,
        tags: Optional[List[str]] = None,
        enabled_only: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> List[RegisteredTool]:
        """
        List registered tools with optional filtering.
        
        Args:
            tags: Filter by tags (any match)
            enabled_only: Only return enabled tools
            limit: Maximum results
            offset: Skip first N results
            
        Returns:
            List of matching tools
        """
        with self._file_lock(exclusive=False):
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT metadata_json, source_code, embedding_json, is_enabled FROM tools"
            conditions = []
            params = []
            
            if enabled_only:
                conditions.append("is_enabled = 1")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            tools = []
            for row in rows:
                tool = RegisteredTool(
                    metadata=ToolMetadata.from_dict(json.loads(row["metadata_json"])),
                    source_code=row["source_code"],
                    embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
                    is_enabled=bool(row["is_enabled"])
                )
                
                if tags:
                    if any(tag in tool.metadata.tags for tag in tags):
                        tools.append(tool)
                else:
                    tools.append(tool)
            
            return tools
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
        min_similarity: float = 0.0
    ) -> List[Tuple[RegisteredTool, float]]:
        """
        Search tools by semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            tags: Optional tag filter
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (tool, similarity_score) tuples
        """
        if self.embedding_fn is None:
            return self._search_keyword(query, top_k, tags)
        
        try:
            query_embedding = self.embedding_fn(query)
        except Exception as e:
            print(f"Warning: Embedding failed, falling back to keyword search: {e}")
            return self._search_keyword(query, top_k, tags)
        
        tools = self.list_tools(tags=tags, enabled_only=True, limit=1000)
        
        scored_tools = []
        for tool in tools:
            if tool.embedding:
                similarity = self._cosine_similarity(query_embedding, tool.embedding)
                if similarity >= min_similarity:
                    scored_tools.append((tool, similarity))
            else:
                keyword_score = self._keyword_score(query, tool)
                if keyword_score > 0:
                    scored_tools.append((tool, keyword_score * 0.5))
        
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools[:top_k]
    
    def _search_keyword(
        self,
        query: str,
        top_k: int,
        tags: Optional[List[str]] = None
    ) -> List[Tuple[RegisteredTool, float]]:
        """Fallback keyword-based search."""
        tools = self.list_tools(tags=tags, enabled_only=True, limit=1000)
        
        scored_tools = []
        for tool in tools:
            score = self._keyword_score(query, tool)
            if score > 0:
                scored_tools.append((tool, score))
        
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools[:top_k]
    
    def _keyword_score(self, query: str, tool: RegisteredTool) -> float:
        """Compute keyword-based relevance score."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        score = 0.0
        
        name_lower = tool.metadata.name.lower()
        if query_lower in name_lower:
            score += 2.0
        for word in query_words:
            if word in name_lower:
                score += 0.5
        
        desc_lower = tool.metadata.description.lower()
        for word in query_words:
            if word in desc_lower:
                score += 0.3
        
        for tag in tool.metadata.tags:
            if tag.lower() in query_lower:
                score += 0.5
        
        return score
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _execute_tool_internal(
        self,
        tool: RegisteredTool,
        inputs: Dict[str, Any]
    ) -> Any:
        """Internal execution without logging."""
        namespace = {"__builtins__": __builtins__}
        
        for module_name in self.allowed_imports:
            try:
                namespace[module_name.split(".")[0]] = __import__(module_name)
            except ImportError:
                pass
        
        exec(tool.source_code, namespace)
        
        if tool.metadata.name not in namespace:
            raise ToolExecutionError(f"Function '{tool.metadata.name}' not found after execution")
        
        func = namespace[tool.metadata.name]
        
        return func(**inputs)
    
    def execute(
        self,
        name: str,
        inputs: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a registered tool.
        
        Args:
            name: Tool name
            inputs: Input parameters
            timeout: Execution timeout (uses default if None)
            
        Returns:
            Tool execution result
            
        Raises:
            ToolNotFoundError: If tool doesn't exist
            ToolExecutionError: If execution fails
        """
        tool = self.get(name)
        
        if not tool.is_enabled:
            raise ToolExecutionError(f"Tool '{name}' is disabled")
        
        timeout = timeout or self.max_execution_time
        start_time = time.time()
        error_msg = None
        result = None
        success = False
        
        try:
            result = self._execute_tool_internal(tool, inputs)
            success = True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            raise ToolExecutionError(f"Execution failed: {error_msg}")
        finally:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            self._log_execution(
                tool_name=name,
                inputs=inputs,
                output=result,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                success=success
            )
        
        return result
    
    def _log_execution(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        output: Any,
        error: Optional[str],
        execution_time_ms: int,
        success: bool
    ):
        """Log tool execution for analytics."""
        try:
            with self._file_lock(exclusive=True):
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO execution_logs (tool_name, input_json, output_json, error, execution_time_ms, success, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    tool_name,
                    json.dumps(inputs, default=str),
                    json.dumps(output, default=str) if output is not None else None,
                    error,
                    execution_time_ms,
                    1 if success else 0,
                    datetime.utcnow().isoformat()
                ))
                
                if success:
                    cursor.execute("""
                        UPDATE tools SET 
                            metadata_json = json_set(metadata_json, '$.usage_count', 
                                COALESCE(json_extract(metadata_json, '$.usage_count'), 0) + 1,
                                '$.success_count', COALESCE(json_extract(metadata_json, '$.success_count'), 0) + 1)
                        WHERE name = ?
                    """, (tool_name,))
                else:
                    cursor.execute("""
                        UPDATE tools SET 
                            metadata_json = json_set(metadata_json, '$.usage_count', 
                                COALESCE(json_extract(metadata_json, '$.usage_count'), 0) + 1,
                                '$.failure_count', COALESCE(json_extract(metadata_json, '$.failure_count'), 0) + 1)
                        WHERE name = ?
                    """, (tool_name,))
                
                conn.commit()
        except Exception as e:
            print(f"Warning: Failed to log execution: {e}")
    
    def disable(self, name: str) -> bool:
        """Disable a tool without deleting it."""
        with self._write_lock:
            with self._file_lock(exclusive=True):
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE tools SET is_enabled = 0 WHERE name = ?", (name,))
                conn.commit()
                return cursor.rowcount > 0
    
    def enable(self, name: str) -> bool:
        """Re-enable a disabled tool."""
        with self._write_lock:
            with self._file_lock(exclusive=True):
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE tools SET is_enabled = 1 WHERE name = ?", (name,))
                conn.commit()
                return cursor.rowcount > 0
    
    def delete(self, name: str) -> bool:
        """Permanently delete a tool (keeps version history)."""
        with self._write_lock:
            with self._file_lock(exclusive=True):
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tools WHERE name = ?", (name,))
                conn.commit()
                return cursor.rowcount > 0
    
    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get usage statistics for a tool."""
        tool = self.get(name)
        
        with self._file_lock(exclusive=False):
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(success) as successful,
                    AVG(execution_time_ms) as avg_time_ms,
                    MIN(created_at) as first_use,
                    MAX(created_at) as last_use
                FROM execution_logs WHERE tool_name = ?
            """, (name,))
            
            row = cursor.fetchone()
            
            return {
                "name": name,
                "version": tool.metadata.version,
                "total_executions": row["total_executions"] or 0,
                "successful_executions": row["successful"] or 0,
                "failure_rate": (
                    (row["total_executions"] - (row["successful"] or 0)) / row["total_executions"]
                    if row["total_executions"] else 0
                ),
                "avg_execution_time_ms": row["avg_time_ms"] or 0,
                "first_use": row["first_use"],
                "last_use": row["last_use"],
                "created_at": tool.metadata.created_at,
                "updated_at": tool.metadata.updated_at
            }
    
    def get_tool_signature(self, name: str) -> str:
        """Get a human-readable signature for a tool."""
        tool = self.get(name)
        m = tool.metadata
        
        params_str = ", ".join(
            f"{pname}: {pinfo.get('type', 'Any')}" + 
            (f" = {pinfo.get('default')}" if 'default' in pinfo else "")
            for pname, pinfo in m.parameters.items()
        )
        
        return f"{m.name}({params_str}) -> {m.return_type}"
    
    def format_tool_for_prompt(self, name: str) -> str:
        """Format tool information for inclusion in an LLM prompt."""
        tool = self.get(name)
        m = tool.metadata
        
        lines = [
            f"[{m.name}]",
            f"  Description: {m.description}",
            f"  Parameters:"
        ]
        
        for pname, pinfo in m.parameters.items():
            required = pinfo.get("required", True)
            default = f", default={pinfo.get('default')}" if 'default' in pinfo else ""
            lines.append(
                f"    - {pname} ({pinfo.get('type', 'Any')}, {'required' if required else 'optional'}{default}): {pinfo.get('description', '')}"
            )
        
        lines.append(f"  Returns: {m.return_type} - {m.return_description}")
        
        if m.tags:
            lines.append(f"  Tags: {', '.join(m.tags)}")
        
        return "\n".join(lines)
    
    def format_all_tools_for_prompt(self, tags: Optional[List[str]] = None) -> str:
        """Format all matching tools for inclusion in an LLM prompt."""
        tools = self.list_tools(tags=tags, enabled_only=True)
        
        if not tools:
            return "No tools available."
        
        formatted = []
        for tool in tools:
            formatted.append(self.format_tool_for_prompt(tool.metadata.name))
        
        return "\n\n".join(formatted)
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Global singleton accessor
_default_registry: Optional[ToolRegistry] = None


def get_registry(
    db_path: str = "tool_registry.db",
    embedding_fn: Optional[Callable[[str], List[float]]] = None
) -> ToolRegistry:
    """Get or create the default tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry(db_path=db_path, embedding_fn=embedding_fn)
    return _default_registry
