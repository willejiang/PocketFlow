"""
Tests for loop detection and action history functionality.

Run with: pytest test_loop_detection.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nodes import (
    _normalize_params,
    _format_action_history,
    _check_duplicate_action,
    MAX_RECENT_ACTIONS,
)


class TestNormalizeParams:
    """Tests for parameter normalization."""
    
    def test_simple_dict(self):
        """Test normalizing a simple dict."""
        params = {"path": "test.py", "content": "hello"}
        normalized = _normalize_params(params)
        assert isinstance(normalized, str)
        # Should be deterministic
        assert _normalize_params(params) == normalized
    
    def test_order_independent(self):
        """Test that order doesn't matter."""
        params1 = {"a": 1, "b": 2}
        params2 = {"b": 2, "a": 1}
        assert _normalize_params(params1) == _normalize_params(params2)
    
    def test_nested_dict(self):
        """Test normalizing nested structures."""
        params = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        normalized = _normalize_params(params)
        assert isinstance(normalized, str)
    
    def test_empty_dict(self):
        """Test normalizing empty dict."""
        assert _normalize_params({}) == "{}"


class TestFormatActionHistory:
    """Tests for action history formatting."""
    
    def test_empty_history(self):
        """Test formatting empty history."""
        result = _format_action_history([])
        assert "No previous actions" in result
    
    def test_single_action(self):
        """Test formatting single action."""
        history = [{
            "action": "read_file",
            "params": {"path": "test.py"},
            "success": True,
            "result_summary": "File contents"
        }]
        result = _format_action_history(history)
        assert "read_file" in result
        assert "test.py" in result
        assert "✓" in result
    
    def test_failed_action(self):
        """Test formatting failed action."""
        history = [{
            "action": "read_file",
            "params": {"path": "missing.py"},
            "success": False,
            "result_summary": "File not found"
        }]
        result = _format_action_history(history)
        assert "✗" in result
        assert "File not found" in result
    
    def test_many_actions_shows_summary(self):
        """Test that many actions shows a summary for older ones."""
        history = []
        for i in range(MAX_RECENT_ACTIONS + 5):
            history.append({
                "action": "read_file",
                "params": {"path": f"file{i}.py"},
                "success": True,
                "result_summary": f"Content {i}"
            })
        
        result = _format_action_history(history)
        # Should have summary of older actions
        assert "Earlier actions" in result
        # Recent ones should be in detail
        assert f"file{MAX_RECENT_ACTIONS + 4}.py" in result
    
    def test_truncates_long_results(self):
        """Test that long results are truncated."""
        history = [{
            "action": "read_file",
            "params": {"path": "big.py"},
            "success": True,
            "result_summary": "x" * 1000  # Very long result
        }]
        result = _format_action_history(history)
        assert "..." in result
        assert len(result) < 1000


class TestCheckDuplicateAction:
    """Tests for duplicate action detection."""
    
    def test_no_duplicate_empty_history(self):
        """Test no duplicate in empty history."""
        result = _check_duplicate_action("read_file", {"path": "test.py"}, [])
        assert result is None
    
    def test_finds_exact_duplicate(self):
        """Test finding exact duplicate."""
        history = [{
            "action": "read_file",
            "params": {"path": "test.py"},
            "params_normalized": _normalize_params({"path": "test.py"}),
            "success": True,
            "result_summary": "Content"
        }]
        result = _check_duplicate_action("read_file", {"path": "test.py"}, history)
        assert result is not None
        assert result["action"] == "read_file"
    
    def test_no_duplicate_different_params(self):
        """Test no duplicate with different params."""
        history = [{
            "action": "read_file",
            "params": {"path": "other.py"},
            "params_normalized": _normalize_params({"path": "other.py"}),
            "success": True,
            "result_summary": "Content"
        }]
        result = _check_duplicate_action("read_file", {"path": "test.py"}, history)
        assert result is None
    
    def test_no_duplicate_different_action(self):
        """Test no duplicate with different action name."""
        history = [{
            "action": "write_file",
            "params": {"path": "test.py"},
            "params_normalized": _normalize_params({"path": "test.py"}),
            "success": True,
            "result_summary": "Written"
        }]
        result = _check_duplicate_action("read_file", {"path": "test.py"}, history)
        assert result is None
    
    def test_lookback_limit(self):
        """Test that lookback limit works."""
        history = []
        # Add the target action at the beginning
        history.append({
            "action": "read_file",
            "params": {"path": "target.py"},
            "params_normalized": _normalize_params({"path": "target.py"}),
            "success": True,
            "result_summary": "Target"
        })
        # Add many other actions after
        for i in range(10):
            history.append({
                "action": "read_file",
                "params": {"path": f"other{i}.py"},
                "params_normalized": _normalize_params({"path": f"other{i}.py"}),
                "success": True,
                "result_summary": f"Other {i}"
            })
        
        # With default lookback of 5, shouldn't find the first one
        result = _check_duplicate_action(
            "read_file", 
            {"path": "target.py"}, 
            history,
            lookback=5
        )
        assert result is None
        
        # With larger lookback, should find it
        result = _check_duplicate_action(
            "read_file",
            {"path": "target.py"},
            history,
            lookback=20
        )
        assert result is not None


class TestDecideActionLoopDetection:
    """Integration tests for DecideAction node.
    
    Note: Loop detection has been disabled. These tests verify basic post() behavior.
    """
    
    def test_post_stores_action_correctly(self):
        """Test that post() correctly stores the action from exec result."""
        from nodes import DecideAction
        
        node = DecideAction(max_iterations=10)
        
        shared = {
            "question": "What is in test.py?",
            "iteration": 0,
            "action_history": [],
            "context": ""
        }
        
        exec_res = {
            "action": "read_file",
            "parameters": {"path": "test.py"},
            "thinking": "Let me read the file"
        }
        
        # prep_res format: (question, history_prompt, context, actions_prompt, iteration)
        prep_res = ("question", "history", "context", "actions", 0)
        result = node.post(shared, prep_res, exec_res)
        
        assert result == "execute"
        assert shared["pending_action"] == "read_file"
        assert shared["pending_parameters"] == {"path": "test.py"}
    
    def test_allows_same_action_multiple_times(self):
        """Test that the same action can be executed multiple times (loop detection disabled)."""
        from nodes import DecideAction
        
        node = DecideAction(max_iterations=10)
        
        action_history = [{
            "action": "read_file",
            "params": {"path": "test.py"},
            "params_normalized": _normalize_params({"path": "test.py"}),
            "success": True,
            "result_summary": "test content"
        }]
        
        shared = {
            "question": "What is in test.py?",
            "iteration": 0,
            "action_history": action_history,
            "context": "Read test.py"
        }
        
        # Same action with same params - should still execute (loop detection disabled)
        exec_res = {
            "action": "read_file",
            "parameters": {"path": "test.py"},
            "thinking": "Reading the file again"
        }
        
        prep_res = ("question", "history", "context", "actions", 0)
        result = node.post(shared, prep_res, exec_res)
        
        # Should return "execute" since loop detection is disabled
        assert result == "execute"
        assert shared["pending_action"] == "read_file"
        assert shared["pending_parameters"] == {"path": "test.py"}


class TestExecuteActionHistoryRecording:
    """Tests for action history recording in ExecuteAction node."""
    
    def test_records_successful_action(self):
        """Test that successful actions are recorded."""
        from nodes import ExecuteAction
        
        node = ExecuteAction()
        
        shared = {
            "pending_action": "read_file",
            "pending_parameters": {"path": "test.py"},
            "action_history": [],
            "context": ""
        }
        
        exec_res = {
            "success": True,
            "result": "file contents",
            "context_update": "Read file: test.py (10 lines)"
        }
        
        prep_res = ("read_file", {"path": "test.py"}, None)
        node.post(shared, prep_res, exec_res)
        
        assert len(shared["action_history"]) == 1
        entry = shared["action_history"][0]
        assert entry["action"] == "read_file"
        assert entry["success"] is True
        assert entry["params"] == {"path": "test.py"}
        assert "params_normalized" in entry
    
    def test_records_failed_action(self):
        """Test that failed actions are recorded."""
        from nodes import ExecuteAction
        
        node = ExecuteAction()
        
        shared = {
            "pending_action": "read_file",
            "pending_parameters": {"path": "missing.py"},
            "action_history": [],
            "context": ""
        }
        
        exec_res = {
            "success": False,
            "error": "File not found"
        }
        
        prep_res = ("read_file", {"path": "missing.py"}, None)
        node.post(shared, prep_res, exec_res)
        
        assert len(shared["action_history"]) == 1
        entry = shared["action_history"][0]
        assert entry["action"] == "read_file"
        assert entry["success"] is False
        assert "File not found" in entry["result_summary"]
    
    def test_handles_empty_success_result(self):
        """Test that empty success results get a descriptive message."""
        from nodes import ExecuteAction
        
        node = ExecuteAction()
        
        shared = {
            "pending_action": "some_action",
            "pending_parameters": {},
            "action_history": [],
            "context": ""
        }
        
        # Empty result
        exec_res = {
            "success": True,
            "result": "",
            "context_update": ""
        }
        
        prep_res = ("some_action", {}, None)
        node.post(shared, prep_res, exec_res)
        
        assert "no output" in shared["context"].lower()
        assert "no output" in shared["action_history"][0]["result_summary"].lower()
    
    def test_handles_none_success_result(self):
        """Test that None success results get a descriptive message."""
        from nodes import ExecuteAction
        
        node = ExecuteAction()
        
        shared = {
            "pending_action": "some_action",
            "pending_parameters": {},
            "action_history": [],
            "context": ""
        }
        
        # None result
        exec_res = {
            "success": True,
            "result": None,
            "context_update": "None"
        }
        
        prep_res = ("some_action", {}, None)
        node.post(shared, prep_res, exec_res)
        
        assert "no output" in shared["context"].lower()
        assert "no output" in shared["action_history"][0]["result_summary"].lower()
    
    def test_handles_empty_error_message(self):
        """Test that empty error messages get a descriptive message."""
        from nodes import ExecuteAction
        
        node = ExecuteAction()
        
        shared = {
            "pending_action": "some_action",
            "pending_parameters": {},
            "action_history": [],
            "context": ""
        }
        
        # Empty error
        exec_res = {
            "success": False,
            "error": ""
        }
        
        prep_res = ("some_action", {}, None)
        node.post(shared, prep_res, exec_res)
        
        assert "no error message" in shared["context"].lower()
        assert "no error message" in shared["action_history"][0]["result_summary"].lower()
    
    def test_adds_loop_reminder_for_duplicate_action(self):
        """Test that duplicate actions get a reminder in context."""
        from nodes import ExecuteAction
        
        node = ExecuteAction()
        
        # First execution already in history
        action_history = [{
            "action": "read_file",
            "params": {"path": "test.py"},
            "params_normalized": _normalize_params({"path": "test.py"}),
            "success": True,
            "result_summary": "file contents here"
        }]
        
        shared = {
            "pending_action": "read_file",
            "pending_parameters": {"path": "test.py"},
            "action_history": action_history,
            "context": ""
        }
        
        # Same action executed again
        exec_res = {
            "success": True,
            "result": "file contents here",
            "context_update": "file contents here"
        }
        
        prep_res = ("read_file", {"path": "test.py"}, None)
        node.post(shared, prep_res, exec_res)
        
        # Should have loop reminder in context
        assert "performed this exact action before" in shared["context"]
        assert "file contents here" in shared["context"]
    
    def test_no_loop_reminder_for_different_params(self):
        """Test that different params don't trigger loop reminder."""
        from nodes import ExecuteAction
        
        node = ExecuteAction()
        
        # First execution with different path
        action_history = [{
            "action": "read_file",
            "params": {"path": "other.py"},
            "params_normalized": _normalize_params({"path": "other.py"}),
            "success": True,
            "result_summary": "other file contents"
        }]
        
        shared = {
            "pending_action": "read_file",
            "pending_parameters": {"path": "test.py"},
            "action_history": action_history,
            "context": ""
        }
        
        # Different file
        exec_res = {
            "success": True,
            "result": "test file contents",
            "context_update": "test file contents"
        }
        
        prep_res = ("read_file", {"path": "test.py"}, None)
        node.post(shared, prep_res, exec_res)
        
        # Should NOT have loop reminder
        assert "performed this exact action before" not in shared["context"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
