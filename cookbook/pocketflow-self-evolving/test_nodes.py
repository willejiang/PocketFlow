"""
Tests for the self-evolving agent nodes.

Tests use mocked LLM responses to avoid API calls.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.tool_registry import ToolRegistry


class TestSafeYamlParse(unittest.TestCase):
    """Test the YAML parsing utility."""
    
    def test_parse_yaml_block(self):
        """Test parsing YAML from markdown code block."""
        from nodes import _safe_yaml_parse
        
        response = '''Here's my response:

```yaml
action: search
query: test query
```

That's all.'''
        
        result = _safe_yaml_parse(response)
        self.assertEqual(result["action"], "search")
        self.assertEqual(result["query"], "test query")
    
    def test_parse_plain_yaml(self):
        """Test parsing plain YAML without code block."""
        from nodes import _safe_yaml_parse
        
        response = '''action: answer
reason: because
answer: 42'''
        
        result = _safe_yaml_parse(response)
        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["answer"], 42)  # YAML parses numbers
    
    def test_parse_invalid_yaml(self):
        """Test handling invalid YAML."""
        from nodes import _safe_yaml_parse
        
        response = "This is not valid YAML at all {"
        result = _safe_yaml_parse(response, {"default": True})
        self.assertEqual(result, {"default": True})
    
    def test_parse_empty(self):
        """Test handling empty input."""
        from nodes import _safe_yaml_parse
        
        result = _safe_yaml_parse("", {"fallback": True})
        self.assertEqual(result, {"fallback": True})
        
        result = _safe_yaml_parse(None, {"fallback": True})
        self.assertEqual(result, {"fallback": True})


class TestToolRegistryIntegration(unittest.TestCase):
    """Test nodes with actual tool registry."""
    
    def setUp(self):
        ToolRegistry._clear_instances()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, f"test_{id(self)}.db")
        self.registry = ToolRegistry(db_path=self.db_path)
        
        # Register test tools
        self.registry.register(
            name="add_numbers",
            description="Add two numbers together",
            source_code="def add_numbers(a, b): return a + b",
            parameters={"a": {"type": "int"}, "b": {"type": "int"}},
            tags=["math", "arithmetic"]
        )
    
    def tearDown(self):
        self.registry.close()
        ToolRegistry._clear_instances()
        for f in [self.db_path, self.db_path + ".lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
    
    def test_search_tools_finds_math(self):
        """Test that search finds relevant tools."""
        from nodes import SearchTools
        
        node = SearchTools()
        shared = {
            "search_query": "add numbers",
            "tool_registry": self.registry  # Pass registry directly
        }
        
        query, registry = node.prep(shared)
        results = node.exec((query, registry))
        
        names = [name for name, _, _ in results]
        self.assertIn("add_numbers", names)
    
    def test_use_tool_executes(self):
        """Test executing a tool."""
        from nodes import UseTool
        
        node = UseTool()
        shared = {
            "execute_tool_name": "add_numbers",
            "execute_tool_inputs": {"a": 5, "b": 3},
            "tool_registry": self.registry  # Pass registry directly
        }
        
        prep_res = node.prep(shared)
        result = node.exec(prep_res)
        
        self.assertEqual(result["result"], 8)
        self.assertIsNone(result["error"])
    
    def test_use_tool_handles_missing(self):
        """Test error handling for missing tool."""
        from nodes import UseTool
        
        node = UseTool()
        shared = {
            "execute_tool_name": "nonexistent",
            "execute_tool_inputs": {},
            "tool_registry": self.registry  # Pass registry directly
        }
        
        prep_res = node.prep(shared)
        result = node.exec(prep_res)
        
        self.assertIsNone(result["result"])
        self.assertIn("not found", result["error"])
    
    def test_list_tools(self):
        """Test listing tools."""
        from nodes import ListTools
        
        node = ListTools()
        shared = {"tool_registry": self.registry}
        
        registry = node.prep(shared)
        tools = node.exec(registry)
        
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "add_numbers")


class TestDecideActionLogic(unittest.TestCase):
    """Test DecideAction node without LLM calls."""
    
    def setUp(self):
        ToolRegistry._clear_instances()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, f"test_{id(self)}.db")
    
    def tearDown(self):
        ToolRegistry._clear_instances()
        for f in [self.db_path, self.db_path + ".lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
    
    def test_post_sets_search_query(self):
        """Test post for search action."""
        from nodes import DecideAction
        
        node = DecideAction()
        shared = {"tool_registry_path": self.db_path}
        exec_res = {"action": "search_tools", "query": "math tools"}
        
        action = node.post(shared, None, exec_res)
        
        self.assertEqual(action, "search_tools")
        self.assertEqual(shared["search_query"], "math tools")
    
    def test_post_sets_tool_name(self):
        """Test post for create_tool action."""
        from nodes import DecideAction
        
        node = DecideAction()
        shared = {"tool_registry_path": self.db_path}
        exec_res = {
            "action": "create_tool",
            "tool_name": "my_tool",
            "description": "Does something"
        }
        
        action = node.post(shared, None, exec_res)
        
        self.assertEqual(action, "create_tool")
        self.assertEqual(shared["new_tool_name"], "my_tool")
    
    def test_post_sets_execute_info(self):
        """Test post for use_tool action."""
        from nodes import DecideAction
        
        node = DecideAction()
        shared = {"tool_registry_path": self.db_path}
        exec_res = {
            "action": "use_tool",
            "tool_name": "add_numbers",
            "inputs": {"a": 1, "b": 2}
        }
        
        action = node.post(shared, None, exec_res)
        
        self.assertEqual(action, "use_tool")
        self.assertEqual(shared["execute_tool_name"], "add_numbers")
        self.assertEqual(shared["execute_tool_inputs"], {"a": 1, "b": 2})
    
    def test_post_sets_answer(self):
        """Test post for answer action."""
        from nodes import DecideAction
        
        node = DecideAction()
        shared = {"tool_registry_path": self.db_path}
        exec_res = {"action": "answer", "answer": "The answer is 42"}
        
        action = node.post(shared, None, exec_res)
        
        self.assertEqual(action, "answer")
        self.assertEqual(shared["final_answer"], "The answer is 42")


if __name__ == "__main__":
    unittest.main(verbosity=2)
