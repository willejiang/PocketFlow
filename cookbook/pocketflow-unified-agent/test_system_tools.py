"""
Tests for the System Tools Adapter.

Run with: pytest test_system_tools.py -v
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path

from adapters import (
    AdapterRegistry,
    get_adapter_registry,
    get_system_tools_adapter,
    SystemToolsAdapter,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def adapter(temp_dir):
    """Create an initialized system tools adapter."""
    adapter = SystemToolsAdapter(
        working_dir=temp_dir,
        allow_commands=True,
        command_timeout=10
    )
    adapter.initialize({})
    return adapter


@pytest.fixture
def registry(adapter):
    """Create a registry with system tools."""
    AdapterRegistry.reset()
    reg = get_adapter_registry()
    reg.register(adapter)
    return reg


class TestSystemToolsAdapter:
    """Tests for SystemToolsAdapter."""
    
    def test_adapter_properties(self, adapter):
        """Test adapter properties."""
        assert adapter.name == "system-tools"
        assert "file" in adapter.description.lower()
        assert "system" in adapter.tags
        assert len(adapter.actions) >= 6  # At least 6 core actions
    
    def test_action_names(self, adapter):
        """Test that all expected actions exist."""
        action_names = [a.name for a in adapter.actions]
        assert "read_file" in action_names
        assert "write_file" in action_names
        assert "list_directory" in action_names
        assert "file_exists" in action_names
        assert "delete_file" in action_names
        assert "create_directory" in action_names
        assert "run_command" in action_names


class TestFileOperations:
    """Tests for file operations."""
    
    def test_write_and_read_file(self, adapter, temp_dir):
        """Test writing and reading a file."""
        test_file = os.path.join(temp_dir, "test.txt")
        content = "Hello, World!"
        
        # Write
        result = adapter.execute("write_file", {
            "path": test_file,
            "content": content
        }, {})
        assert result["success"]
        
        # Read
        result = adapter.execute("read_file", {"path": test_file}, {})
        assert result["success"]
        assert result["result"] == content
    
    def test_read_file_line_range(self, adapter, temp_dir):
        """Test reading specific lines."""
        test_file = os.path.join(temp_dir, "lines.txt")
        content = "\n".join([f"Line {i}" for i in range(1, 11)])
        
        # Write
        adapter.execute("write_file", {"path": test_file, "content": content}, {})
        
        # Read lines 3-5
        result = adapter.execute("read_file", {
            "path": test_file,
            "start_line": 3,
            "end_line": 5
        }, {})
        assert result["success"]
        lines = result["result"].strip().split("\n")
        assert len(lines) == 3
        assert "Line 3" in lines[0]
    
    def test_read_nonexistent_file(self, adapter, temp_dir):
        """Test reading a file that doesn't exist."""
        result = adapter.execute("read_file", {
            "path": os.path.join(temp_dir, "nonexistent.txt")
        }, {})
        assert not result["success"]
        assert "not found" in result["error"].lower()
    
    def test_append_to_file(self, adapter, temp_dir):
        """Test appending to a file."""
        test_file = os.path.join(temp_dir, "append.txt")
        
        # Write initial content
        adapter.execute("write_file", {
            "path": test_file,
            "content": "First line\n"
        }, {})
        
        # Append
        result = adapter.execute("write_file", {
            "path": test_file,
            "content": "Second line\n",
            "append": True
        }, {})
        assert result["success"]
        
        # Verify
        result = adapter.execute("read_file", {"path": test_file}, {})
        assert "First line" in result["result"]
        assert "Second line" in result["result"]
    
    def test_file_exists(self, adapter, temp_dir):
        """Test file existence check."""
        test_file = os.path.join(temp_dir, "exists.txt")
        
        # Doesn't exist yet
        result = adapter.execute("file_exists", {"path": test_file}, {})
        assert result["success"]
        assert result["result"] is False
        
        # Create it
        adapter.execute("write_file", {"path": test_file, "content": "test"}, {})
        
        # Now exists
        result = adapter.execute("file_exists", {"path": test_file}, {})
        assert result["success"]
        assert result["result"] is True
    
    def test_delete_file(self, adapter, temp_dir):
        """Test file deletion."""
        test_file = os.path.join(temp_dir, "delete_me.txt")
        
        # Create
        adapter.execute("write_file", {"path": test_file, "content": "delete me"}, {})
        
        # Delete
        result = adapter.execute("delete_file", {"path": test_file}, {})
        assert result["success"]
        
        # Verify deleted
        result = adapter.execute("file_exists", {"path": test_file}, {})
        assert result["result"] is False


class TestDirectoryOperations:
    """Tests for directory operations."""
    
    def test_create_directory(self, adapter, temp_dir):
        """Test directory creation."""
        test_dir = os.path.join(temp_dir, "new_dir", "nested")
        
        result = adapter.execute("create_directory", {"path": test_dir}, {})
        assert result["success"]
        assert os.path.isdir(test_dir)
    
    def test_list_directory(self, adapter, temp_dir):
        """Test directory listing."""
        # Create some files
        for i in range(3):
            adapter.execute("write_file", {
                "path": os.path.join(temp_dir, f"file{i}.txt"),
                "content": f"content {i}"
            }, {})
        
        result = adapter.execute("list_directory", {"path": temp_dir}, {})
        assert result["success"]
        assert len(result["result"]) == 3
    
    def test_list_directory_with_pattern(self, adapter, temp_dir):
        """Test directory listing with glob pattern."""
        # Create mixed files
        adapter.execute("write_file", {"path": os.path.join(temp_dir, "a.txt"), "content": ""}, {})
        adapter.execute("write_file", {"path": os.path.join(temp_dir, "b.txt"), "content": ""}, {})
        adapter.execute("write_file", {"path": os.path.join(temp_dir, "c.py"), "content": ""}, {})
        
        # List only .txt files
        result = adapter.execute("list_directory", {
            "path": temp_dir,
            "pattern": "*.txt"
        }, {})
        assert result["success"]
        assert len(result["result"]) == 2


class TestCommandExecution:
    """Tests for command execution."""
    
    def test_run_simple_command(self, adapter):
        """Test running a simple command."""
        result = adapter.execute("run_command", {
            "command": "echo 'Hello World'"
        }, {})
        assert result["success"]
        assert "Hello World" in result["result"]["stdout"]
    
    def test_run_command_with_exit_code(self, adapter):
        """Test command with non-zero exit code."""
        result = adapter.execute("run_command", {
            "command": "exit 1"
        }, {})
        assert not result["success"]
        assert result["result"]["returncode"] == 1
    
    def test_run_command_with_stderr(self, adapter):
        """Test command that produces stderr."""
        result = adapter.execute("run_command", {
            "command": "echo 'error' >&2"
        }, {})
        assert result["success"]  # Exit code 0
        assert "error" in result["result"]["stderr"]
    
    def test_run_command_disabled(self, temp_dir):
        """Test that commands are disabled when configured."""
        adapter = get_system_tools_adapter(
            working_dir=temp_dir,
            allow_commands=False
        )
        adapter.initialize({})
        
        result = adapter.execute("run_command", {"command": "echo test"}, {})
        assert not result["success"]
        assert "disabled" in result["error"].lower()


class TestRegistryIntegration:
    """Tests for registry integration."""
    
    def test_registry_execute_action(self, registry, temp_dir):
        """Test executing actions through the registry."""
        test_file = os.path.join(temp_dir, "registry_test.txt")
        
        result = registry.execute_action("write_file", {
            "path": test_file,
            "content": "via registry"
        }, {})
        assert result["success"]
        
        result = registry.execute_action("read_file", {"path": test_file}, {})
        assert result["success"]
        assert result["result"] == "via registry"
    
    def test_registry_list_actions(self, registry):
        """Test listing all actions from registry."""
        actions = registry.list_all_actions()
        action_names = [a.name for _, a in actions]
        
        assert "read_file" in action_names
        assert "write_file" in action_names
        assert "run_command" in action_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
