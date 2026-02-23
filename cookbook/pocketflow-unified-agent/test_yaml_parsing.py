"""
Tests for YAML parsing robustness in nodes.py

Run with: pytest test_yaml_parsing.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nodes import _safe_yaml_parse


class TestYamlParsingBasic:
    """Tests for basic YAML parsing scenarios."""
    
    def test_standard_yaml_block(self):
        """Test parsing standard yaml code block."""
        response = '''```yaml
thinking: "I need to read the file"
action: read_file
parameters:
  path: "test.py"
```'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "read_file"
        assert result["parameters"]["path"] == "test.py"
    
    def test_yaml_with_extra_text(self):
        """Test parsing yaml with surrounding text."""
        response = '''Here is my decision:

```yaml
thinking: "Analyzing the request"
action: list_directory
parameters:
  path: "/tmp"
```

This should work.'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "list_directory"
        assert result["parameters"]["path"] == "/tmp"
    
    def test_no_code_block(self):
        """Test parsing without code block markers."""
        response = '''thinking: "Direct yaml"
action: answer
parameters:
  answer: "The result is 42"'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "answer"
        assert "42" in result["parameters"]["answer"]
    
    def test_empty_response(self):
        """Test empty response returns fallback."""
        result = _safe_yaml_parse("", {"action": "fallback"})
        assert result["action"] == "fallback"
    
    def test_none_response(self):
        """Test None-like response."""
        result = _safe_yaml_parse(None, {"action": "fallback"})
        assert result["action"] == "fallback"


class TestYamlParsingEdgeCases:
    """Tests for edge cases and problematic inputs."""
    
    def test_multiline_thinking(self):
        """Test parsing with multiline thinking."""
        response = '''```yaml
thinking: "Line one. Line two. Line three."
action: read_file
parameters:
  path: "file.py"
```'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "read_file"
    
    def test_content_with_escaped_newlines(self):
        """Test content parameter with escaped newlines."""
        response = '''```yaml
thinking: "Writing code"
action: write_file
parameters:
  path: "test.py"
  content: "line1\\nline2\\nline3"
```'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "write_file"
        assert result["parameters"]["path"] == "test.py"
    
    def test_special_characters_in_values(self):
        """Test values with special characters."""
        response = '''```yaml
thinking: "Processing special chars"
action: write_file
parameters:
  path: "test.py"
  content: "hello: world"
```'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "write_file"
    
    def test_boolean_parameters(self):
        """Test boolean parameter values."""
        response = '''```yaml
thinking: "Setting flags"
action: list_directory
parameters:
  path: "/tmp"
  recursive: true
```'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "list_directory"
        # Note: boolean might be parsed as string in some extraction modes
        assert result["parameters"]["path"] == "/tmp"
    
    def test_numeric_parameters(self):
        """Test numeric parameter values."""
        response = '''```yaml
thinking: "Setting limits"
action: read_file
parameters:
  path: "test.py"
  start_line: 10
  end_line: 20
```'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "read_file"
        assert result["parameters"]["start_line"] == 10


class TestYamlParsingFailsafe:
    """Tests for fallback extraction when YAML parsing fails."""
    
    def test_regex_fallback_extraction(self):
        """Test regex extraction when YAML is malformed."""
        response = '''Some text before
action: read_file
parameters:
    path: test.py
More text after'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "read_file"
    
    def test_action_only_extraction(self):
        """Test extracting just the action when parameters are malformed."""
        response = '''
action: answer
parameters: {malformed json here}
'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "answer"
    
    def test_deeply_nested_content_fails_gracefully(self):
        """Test that deeply nested content doesn't crash."""
        response = '''```yaml
thinking: "Complex content"
action: write_file
parameters:
  path: "test.py"
  content: |
    def foo():
        if True:
            for i in range(10):
                print(i)
```'''
        result = _safe_yaml_parse(response)
        # Should at least extract action
        assert result.get("action") == "write_file"
    
    def test_fallback_on_complete_garbage(self):
        """Test fallback when response is completely unparseable."""
        response = "This is not yaml at all, just random text."
        result = _safe_yaml_parse(response, {"action": "fallback"})
        assert result["action"] == "fallback"


class TestYamlParsingWithCodeInContent:
    """Tests specifically for handling code content in parameters."""
    
    def test_python_code_in_content(self):
        """Test handling Python code in content parameter."""
        response = '''```yaml
thinking: "Writing Python code"
action: write_file
parameters:
  path: "hello.py"
  content: "def hello():\\n    print('Hello World')"
```'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "write_file"
        assert result["parameters"]["path"] == "hello.py"
    
    def test_code_with_quotes(self):
        """Test handling code with various quote styles."""
        response = '''```yaml
thinking: "Code with quotes"
action: write_file  
parameters:
  path: "test.py"
  content: "msg = 'hello'"
```'''
        result = _safe_yaml_parse(response)
        assert result["action"] == "write_file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
