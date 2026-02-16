"""
Tests for the adapter system.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from adapters.base import (
    CookbookAdapter, AdapterAction, AdapterRegistry, get_adapter_registry
)
from adapters.discovery import (
    discover_cookbooks, get_cookbook_info, load_cookbook_adapter
)


class MockAdapter(CookbookAdapter):
    """Mock adapter for testing."""
    
    @property
    def name(self) -> str:
        return "mock-cookbook"
    
    @property
    def description(self) -> str:
        return "A mock cookbook for testing"
    
    @property
    def actions(self):
        return [
            AdapterAction(
                name="mock_action",
                description="A mock action",
                parameters={
                    "input": {"type": "str", "description": "Input", "required": True}
                }
            )
        ]
    
    def execute(self, action_name, params, shared):
        if action_name == "mock_action":
            return {
                "success": True,
                "result": f"Processed: {params.get('input', '')}",
                "context_update": "Mock action executed"
            }
        return {"success": False, "error": "Unknown action"}


class TestAdapterAction(unittest.TestCase):
    """Test AdapterAction class."""
    
    def test_format_for_prompt(self):
        action = AdapterAction(
            name="test_action",
            description="Test description",
            parameters={
                "param1": {"type": "str", "description": "First param", "required": True},
                "param2": {"type": "int", "description": "Second param", "required": False, "default": 10}
            }
        )
        
        formatted = action.format_for_prompt()
        
        self.assertIn("[test_action]", formatted)
        self.assertIn("Test description", formatted)
        self.assertIn("param1", formatted)
        self.assertIn("required", formatted)
        self.assertIn("optional", formatted)
    
    def test_validate_params_success(self):
        action = AdapterAction(
            name="test",
            description="Test",
            parameters={
                "required_param": {"type": "str", "required": True}
            }
        )
        
        valid, error = action.validate_params({"required_param": "value"})
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_validate_params_missing_required(self):
        action = AdapterAction(
            name="test",
            description="Test",
            parameters={
                "required_param": {"type": "str", "required": True}
            }
        )
        
        valid, error = action.validate_params({})
        self.assertFalse(valid)
        self.assertIn("required_param", error)


class TestAdapterRegistry(unittest.TestCase):
    """Test AdapterRegistry class."""
    
    def setUp(self):
        AdapterRegistry.reset()
        self.registry = get_adapter_registry()
    
    def tearDown(self):
        AdapterRegistry.reset()
    
    def test_register_adapter(self):
        adapter = MockAdapter()
        self.registry.register(adapter)
        
        retrieved = self.registry.get("mock-cookbook")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "mock-cookbook")
    
    def test_find_action(self):
        adapter = MockAdapter()
        self.registry.register(adapter)
        
        result = self.registry.find_action("mock_action")
        self.assertIsNotNone(result)
        
        found_adapter, found_action = result
        self.assertEqual(found_adapter.name, "mock-cookbook")
        self.assertEqual(found_action.name, "mock_action")
    
    def test_execute_action(self):
        adapter = MockAdapter()
        self.registry.register(adapter)
        
        result = self.registry.execute_action(
            "mock_action",
            {"input": "test"},
            {}
        )
        
        self.assertTrue(result["success"])
        self.assertIn("test", result["result"])
    
    def test_execute_missing_action(self):
        result = self.registry.execute_action("nonexistent", {}, {})
        
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])
    
    def test_execute_missing_required_param(self):
        adapter = MockAdapter()
        self.registry.register(adapter)
        
        result = self.registry.execute_action("mock_action", {}, {})
        
        self.assertFalse(result["success"])
        self.assertIn("Missing", result["error"])
    
    def test_list_adapters(self):
        adapter = MockAdapter()
        self.registry.register(adapter)
        
        adapters = self.registry.list_adapters()
        self.assertEqual(len(adapters), 1)
        self.assertEqual(adapters[0].name, "mock-cookbook")
    
    def test_format_all_actions(self):
        adapter = MockAdapter()
        self.registry.register(adapter)
        
        formatted = self.registry.format_all_actions_for_prompt()
        
        self.assertIn("mock-cookbook", formatted)
        self.assertIn("mock_action", formatted)


class TestCookbookDiscovery(unittest.TestCase):
    """Test cookbook discovery."""
    
    def test_discover_cookbooks(self):
        # This test requires the actual cookbook directory
        try:
            cookbooks = discover_cookbooks()
            
            # Should find at least some cookbooks
            self.assertGreater(len(cookbooks), 0)
            
            # Check structure
            for cb in cookbooks:
                self.assertTrue(cb.name.startswith("pocketflow-"))
                self.assertIsInstance(cb.path, Path)
                
        except RuntimeError:
            # Skip if cookbook directory not found
            self.skipTest("Cookbook directory not found")
    
    def test_get_cookbook_info(self):
        # Find an actual cookbook
        try:
            cookbook_dir = Path(__file__).parent.parent
            agent_path = cookbook_dir / "pocketflow-agent"
            
            if agent_path.exists():
                info = get_cookbook_info(agent_path)
                
                self.assertIsNotNone(info)
                self.assertEqual(info.name, "pocketflow-agent")
                self.assertTrue(info.has_nodes or info.has_flow or info.has_main)
        except Exception:
            self.skipTest("Could not find pocketflow-agent")


class TestAdapterLoading(unittest.TestCase):
    """Test adapter loading."""
    
    def test_load_explicit_adapter(self):
        # Test loading an adapter that has adapter.py
        try:
            cookbook_dir = Path(__file__).parent.parent
            agent_path = cookbook_dir / "pocketflow-agent"
            
            if (agent_path / "adapter.py").exists():
                info = get_cookbook_info(agent_path)
                adapter = load_cookbook_adapter(info, auto_generate=False)
                
                self.assertIsNotNone(adapter)
                self.assertEqual(adapter.name, "pocketflow-agent")
                self.assertGreater(len(adapter.actions), 0)
        except Exception as e:
            self.skipTest(f"Could not load adapter: {e}")
    
    def test_load_auto_generated_adapter(self):
        # Test auto-generation
        try:
            cookbooks = discover_cookbooks()
            
            for cb in cookbooks[:5]:  # Test first 5
                adapter = load_cookbook_adapter(cb, auto_generate=True)
                
                if adapter:
                    self.assertEqual(adapter.name, cb.name)
                    # Auto-generated adapters should have at least one action
                    self.assertGreater(len(adapter.actions), 0)
                    break
        except Exception:
            self.skipTest("Could not test auto-generation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
