"""
Tests for the Tool Registry.

Tests cover:
- Tool registration and validation
- Tool retrieval and search
- Tool execution
- Edge cases and error handling
- Concurrency safety
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from tools.tool_registry import (
    ToolRegistry,
    RegisteredTool,
    ToolMetadata,
    ToolValidationError,
    ToolExecutionError,
    ToolNotFoundError,
    get_registry
)


class TestToolRegistry(unittest.TestCase):
    """Test cases for ToolRegistry."""
    
    def setUp(self):
        """Create a temporary database for each test."""
        # Clear singleton instances to ensure clean state
        ToolRegistry._clear_instances()
        
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, f"test_registry_{id(self)}.db")
        self.registry = ToolRegistry(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.registry.close()
        
        # Clear singleton instances
        ToolRegistry._clear_instances()
        
        # Clean up files
        for f in [self.db_path, self.db_path + ".lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass
    
    def test_register_simple_tool(self):
        """Test registering a simple tool."""
        source = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        tool = self.registry.register(
            name="add_numbers",
            description="Adds two numbers together",
            source_code=source,
            parameters={
                "a": {"type": "int", "description": "First number", "required": True},
                "b": {"type": "int", "description": "Second number", "required": True}
            },
            return_type="int",
            return_description="Sum of a and b"
        )
        
        self.assertEqual(tool.metadata.name, "add_numbers")
        self.assertEqual(tool.metadata.version, 1)
        self.assertTrue(tool.is_enabled)
    
    def test_register_with_examples(self):
        """Test registering a tool with test examples."""
        source = '''
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
'''
        examples = [
            {"input": {"a": 2, "b": 3}, "output": 6},
            {"input": {"a": 0, "b": 100}, "output": 0},
            {"input": {"a": -1, "b": 5}, "output": -5}
        ]
        
        tool = self.registry.register(
            name="multiply",
            description="Multiplies two numbers",
            source_code=source,
            parameters={
                "a": {"type": "int", "description": "First number"},
                "b": {"type": "int", "description": "Second number"}
            },
            return_type="int",
            examples=examples,
            test_examples=True
        )
        
        self.assertEqual(tool.metadata.name, "multiply")
        self.assertEqual(len(tool.metadata.examples), 3)
    
    def test_register_fails_with_bad_example(self):
        """Test that registration fails when examples don't pass."""
        source = '''
def bad_multiply(a: int, b: int) -> int:
    """This implementation is wrong."""
    return a + b  # Wrong!
'''
        examples = [
            {"input": {"a": 2, "b": 3}, "output": 6}  # Expects 6, will get 5
        ]
        
        with self.assertRaises(ToolValidationError):
            self.registry.register(
                name="bad_multiply",
                description="A broken multiply",
                source_code=source,
                parameters={"a": {}, "b": {}},
                examples=examples,
                test_examples=True
            )
    
    def test_register_fails_with_syntax_error(self):
        """Test that registration fails with syntax errors."""
        source = '''
def broken_func(a: int
    return a  # Missing closing paren
'''
        with self.assertRaises(ToolValidationError):
            self.registry.register(
                name="broken_func",
                description="Broken",
                source_code=source,
                parameters={}
            )
    
    def test_register_fails_with_dangerous_code(self):
        """Test that dangerous code patterns are rejected."""
        dangerous_sources = [
            ('def evil(x): return eval(x)', "eval"),
            ('def evil(x): exec(x)', "exec"),
            ('def evil(): import os; return os.system("ls")', "os.system"),
            ('def evil(): return open("/etc/passwd").read()', "open"),
        ]
        
        for source, pattern in dangerous_sources:
            with self.assertRaises(ToolValidationError, msg=f"Should reject {pattern}"):
                self.registry.register(
                    name="evil",
                    description="Evil tool",
                    source_code=source,
                    parameters={}
                )
    
    def test_register_fails_with_wrong_function_name(self):
        """Test that function name must match tool name."""
        source = '''
def wrong_name(x: int) -> int:
    return x * 2
'''
        with self.assertRaises(ToolValidationError):
            self.registry.register(
                name="correct_name",
                description="Name mismatch",
                source_code=source,
                parameters={}
            )
    
    def test_get_tool(self):
        """Test retrieving a registered tool."""
        source = '''
def get_test(x: int) -> int:
    return x * 2
'''
        self.registry.register(
            name="get_test",
            description="Test",
            source_code=source,
            parameters={"x": {"type": "int"}}
        )
        
        tool = self.registry.get("get_test")
        self.assertEqual(tool.metadata.name, "get_test")
        self.assertIn("def get_test", tool.source_code)
    
    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        with self.assertRaises(ToolNotFoundError):
            self.registry.get("nonexistent_tool")
    
    def test_exists(self):
        """Test checking if a tool exists."""
        source = 'def exists_test(x): return x'
        
        self.assertFalse(self.registry.exists("exists_test"))
        
        self.registry.register(
            name="exists_test",
            description="Test",
            source_code=source,
            parameters={}
        )
        
        self.assertTrue(self.registry.exists("exists_test"))
    
    def test_list_tools(self):
        """Test listing tools."""
        # Register several tools
        for i in range(5):
            source = f'def tool_{i}(x): return x + {i}'
            self.registry.register(
                name=f"tool_{i}",
                description=f"Tool number {i}",
                source_code=source,
                parameters={"x": {"type": "int"}},
                tags=["test", f"number_{i}"]
            )
        
        # List all
        tools = self.registry.list_tools()
        self.assertEqual(len(tools), 5)
        
        # List with limit
        tools = self.registry.list_tools(limit=3)
        self.assertEqual(len(tools), 3)
        
        # List with tag filter
        tools = self.registry.list_tools(tags=["number_2"])
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].metadata.name, "tool_2")
    
    def test_execute_tool(self):
        """Test executing a registered tool."""
        source = '''
def exec_test(a: int, b: int) -> int:
    """Add numbers."""
    return a + b
'''
        self.registry.register(
            name="exec_test",
            description="Add",
            source_code=source,
            parameters={"a": {}, "b": {}}
        )
        
        result = self.registry.execute("exec_test", {"a": 5, "b": 3})
        self.assertEqual(result, 8)
    
    def test_execute_with_allowed_imports(self):
        """Test that allowed imports work in tools."""
        source = '''
import math

def math_test(x: float) -> float:
    """Calculate square root."""
    return math.sqrt(x)
'''
        self.registry.register(
            name="math_test",
            description="Square root",
            source_code=source,
            parameters={"x": {"type": "float"}}
        )
        
        result = self.registry.execute("math_test", {"x": 16.0})
        self.assertEqual(result, 4.0)
    
    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist."""
        with self.assertRaises(ToolNotFoundError):
            self.registry.execute("nonexistent", {})
    
    def test_execute_disabled_tool(self):
        """Test that disabled tools can't be executed."""
        source = 'def disabled_test(x): return x'
        self.registry.register(
            name="disabled_test",
            description="Test",
            source_code=source,
            parameters={}
        )
        
        self.registry.disable("disabled_test")
        
        with self.assertRaises(ToolExecutionError):
            self.registry.execute("disabled_test", {"x": 1})
    
    def test_version_increment(self):
        """Test that updating a tool increments version."""
        source_v1 = 'def version_test(x): return x'
        source_v2 = 'def version_test(x): return x * 2'
        
        tool_v1 = self.registry.register(
            name="version_test",
            description="Version 1",
            source_code=source_v1,
            parameters={}
        )
        self.assertEqual(tool_v1.metadata.version, 1)
        
        tool_v2 = self.registry.register(
            name="version_test",
            description="Version 2",
            source_code=source_v2,
            parameters={}
        )
        self.assertEqual(tool_v2.metadata.version, 2)
    
    def test_get_specific_version(self):
        """Test retrieving a specific version of a tool."""
        source_v1 = 'def ver_test(x): return x'
        source_v2 = 'def ver_test(x): return x * 2'
        
        self.registry.register(
            name="ver_test",
            description="V1",
            source_code=source_v1,
            parameters={}
        )
        self.registry.register(
            name="ver_test",
            description="V2",
            source_code=source_v2,
            parameters={}
        )
        
        # Get latest
        latest = self.registry.get("ver_test")
        self.assertEqual(latest.metadata.version, 2)
        
        # Get v1
        v1 = self.registry.get("ver_test", version=1)
        self.assertEqual(v1.metadata.version, 1)
        self.assertIn("return x", v1.source_code)
        self.assertNotIn("* 2", v1.source_code)
    
    def test_disable_enable(self):
        """Test disabling and re-enabling tools."""
        source = 'def toggle_test(x): return x'
        self.registry.register(
            name="toggle_test",
            description="Test",
            source_code=source,
            parameters={}
        )
        
        # Initially enabled
        tool = self.registry.get("toggle_test")
        self.assertTrue(tool.is_enabled)
        
        # Disable
        self.registry.disable("toggle_test")
        tool = self.registry.get("toggle_test")
        self.assertFalse(tool.is_enabled)
        
        # Re-enable
        self.registry.enable("toggle_test")
        tool = self.registry.get("toggle_test")
        self.assertTrue(tool.is_enabled)
    
    def test_delete(self):
        """Test deleting a tool."""
        source = 'def delete_test(x): return x'
        self.registry.register(
            name="delete_test",
            description="Test",
            source_code=source,
            parameters={}
        )
        
        self.assertTrue(self.registry.exists("delete_test"))
        
        self.registry.delete("delete_test")
        
        self.assertFalse(self.registry.exists("delete_test"))
    
    def test_keyword_search(self):
        """Test keyword-based search."""
        # Register tools with different descriptions
        tools_data = [
            ("calc_sum", "Calculate the sum of numbers"),
            ("calc_product", "Calculate the product of numbers"),
            ("format_text", "Format text with various options"),
            ("parse_json", "Parse JSON string into object"),
        ]
        
        for name, desc in tools_data:
            source = f'def {name}(x): return x'
            self.registry.register(
                name=name,
                description=desc,
                source_code=source,
                parameters={"x": {}}
            )
        
        # Search for "calculate"
        results = self.registry.search("calculate", top_k=10)
        names = [t.metadata.name for t, _ in results]
        self.assertIn("calc_sum", names)
        self.assertIn("calc_product", names)
        
        # Search for "json"
        results = self.registry.search("json", top_k=10)
        names = [t.metadata.name for t, _ in results]
        self.assertIn("parse_json", names)
    
    def test_get_stats(self):
        """Test getting tool statistics."""
        source = 'def stats_test(x): return x * 2'
        self.registry.register(
            name="stats_test",
            description="Test",
            source_code=source,
            parameters={"x": {"type": "int"}}
        )
        
        # Execute a few times
        for i in range(3):
            self.registry.execute("stats_test", {"x": i})
        
        stats = self.registry.get_stats("stats_test")
        self.assertEqual(stats["name"], "stats_test")
        self.assertEqual(stats["total_executions"], 3)
        self.assertEqual(stats["successful_executions"], 3)
    
    def test_format_tool_for_prompt(self):
        """Test formatting tool info for LLM prompts."""
        source = '''
def format_test(text: str, max_length: int = 100) -> str:
    """Truncate text to max length."""
    return text[:max_length]
'''
        self.registry.register(
            name="format_test",
            description="Truncates text to a maximum length",
            source_code=source,
            parameters={
                "text": {"type": "str", "description": "Input text", "required": True},
                "max_length": {"type": "int", "description": "Maximum length", "required": False, "default": 100}
            },
            return_type="str",
            return_description="Truncated text",
            tags=["text", "formatting"]
        )
        
        formatted = self.registry.format_tool_for_prompt("format_test")
        
        self.assertIn("[format_test]", formatted)
        self.assertIn("Truncates text", formatted)
        self.assertIn("text (str", formatted)
        self.assertIn("max_length (int", formatted)
        self.assertIn("optional", formatted)
        self.assertIn("default=100", formatted)
    
    def test_concurrent_registration(self):
        """Test thread-safe concurrent registration."""
        errors = []
        
        def register_tool(i):
            try:
                source = f'def concurrent_{i}(x): return x + {i}'
                self.registry.register(
                    name=f"concurrent_{i}",
                    description=f"Concurrent tool {i}",
                    source_code=source,
                    parameters={"x": {"type": "int"}}
                )
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=register_tool, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0, f"Errors during concurrent registration: {errors}")
        
        # Verify all tools exist
        tools = self.registry.list_tools()
        self.assertEqual(len(tools), 10)
    
    def test_concurrent_execution(self):
        """Test thread-safe concurrent execution."""
        source = '''
def concurrent_exec(x: int) -> int:
    return x * 2
'''
        self.registry.register(
            name="concurrent_exec",
            description="Test",
            source_code=source,
            parameters={"x": {"type": "int"}}
        )
        
        results = []
        errors = []
        
        def execute_tool(x):
            try:
                result = self.registry.execute("concurrent_exec", {"x": x})
                results.append((x, result))
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=execute_tool, args=(i,)) for i in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0, f"Errors during concurrent execution: {errors}")
        self.assertEqual(len(results), 20)
        
        # Verify results
        for x, result in results:
            self.assertEqual(result, x * 2)


class TestToolMetadata(unittest.TestCase):
    """Test cases for ToolMetadata."""
    
    def test_to_dict_from_dict(self):
        """Test serialization/deserialization."""
        metadata = ToolMetadata(
            name="test_tool",
            description="A test tool",
            parameters={"x": {"type": "int"}},
            return_type="int",
            return_description="The result",
            tags=["test", "example"],
            examples=[{"input": {"x": 1}, "output": 2}]
        )
        
        data = metadata.to_dict()
        restored = ToolMetadata.from_dict(data)
        
        self.assertEqual(restored.name, metadata.name)
        self.assertEqual(restored.description, metadata.description)
        self.assertEqual(restored.parameters, metadata.parameters)
        self.assertEqual(restored.tags, metadata.tags)


class TestRegisteredTool(unittest.TestCase):
    """Test cases for RegisteredTool."""
    
    def test_to_dict_from_dict(self):
        """Test serialization/deserialization."""
        metadata = ToolMetadata(
            name="test",
            description="Test",
            parameters={},
            return_type="Any",
            return_description=""
        )
        
        tool = RegisteredTool(
            metadata=metadata,
            source_code="def test(): pass",
            embedding=[0.1, 0.2, 0.3],
            is_enabled=True
        )
        
        data = tool.to_dict()
        restored = RegisteredTool.from_dict(data)
        
        self.assertEqual(restored.metadata.name, tool.metadata.name)
        self.assertEqual(restored.source_code, tool.source_code)
        self.assertEqual(restored.embedding, tool.embedding)
        self.assertEqual(restored.is_enabled, tool.is_enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
