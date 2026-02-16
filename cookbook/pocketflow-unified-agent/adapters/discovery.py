"""
Cookbook Discovery System

Automatically discovers cookbooks and their adapters.
Supports multiple ways to expose capabilities:
1. Explicit adapter.py file in the cookbook
2. cookbook_manifest.yaml describing the cookbook
3. Auto-generated adapter from analyzing nodes.py/flow.py/main.py
"""

import os
import sys
import re
import ast
import yaml
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass

from .base import CookbookAdapter, AdapterAction


@dataclass
class CookbookInfo:
    """Information about a discovered cookbook."""
    name: str
    path: Path
    title: str
    description: str
    has_adapter: bool
    has_manifest: bool
    has_nodes: bool
    has_flow: bool
    has_main: bool
    dependencies: List[str]
    tags: List[str]


def get_cookbook_dir() -> Path:
    """Get the cookbook directory path."""
    # Try relative to this file first
    current = Path(__file__).parent.parent.parent
    if (current / "pocketflow-agent").exists():
        return current
    
    # Try environment variable
    if "POCKETFLOW_COOKBOOK" in os.environ:
        return Path(os.environ["POCKETFLOW_COOKBOOK"])
    
    # Try current directory
    if Path("pocketflow-agent").exists():
        return Path(".")
    
    raise RuntimeError("Could not find cookbook directory")


def discover_cookbooks(cookbook_dir: Optional[Path] = None) -> List[CookbookInfo]:
    """
    Discover all cookbooks in the cookbook directory.
    
    Returns list of CookbookInfo objects describing each cookbook.
    """
    if cookbook_dir is None:
        cookbook_dir = get_cookbook_dir()
    
    cookbooks = []
    
    for item in cookbook_dir.iterdir():
        if not item.is_dir():
            continue
        
        if not item.name.startswith("pocketflow-"):
            continue
        
        # Skip the unified agent itself
        if item.name == "pocketflow-unified-agent":
            continue
        
        info = get_cookbook_info(item)
        if info:
            cookbooks.append(info)
    
    # Sort by name
    cookbooks.sort(key=lambda x: x.name)
    
    return cookbooks


def get_cookbook_info(cookbook_path: Path) -> Optional[CookbookInfo]:
    """Get information about a single cookbook."""
    if not cookbook_path.is_dir():
        return None
    
    name = cookbook_path.name
    
    # Check for various files
    has_adapter = (cookbook_path / "adapter.py").exists()
    has_manifest = (cookbook_path / "cookbook_manifest.yaml").exists()
    has_nodes = (cookbook_path / "nodes.py").exists()
    has_flow = (cookbook_path / "flow.py").exists()
    has_main = (cookbook_path / "main.py").exists()
    
    # Get title and description from README
    title = name
    description = ""
    
    readme_path = cookbook_path / "README.md"
    if readme_path.exists():
        title, description = _parse_readme(readme_path)
    
    # Get dependencies from requirements.txt
    dependencies = []
    req_path = cookbook_path / "requirements.txt"
    if req_path.exists():
        dependencies = _parse_requirements(req_path)
    
    # Get tags from manifest or infer from name
    tags = _infer_tags(name, cookbook_path)
    
    return CookbookInfo(
        name=name,
        path=cookbook_path,
        title=title,
        description=description,
        has_adapter=has_adapter,
        has_manifest=has_manifest,
        has_nodes=has_nodes,
        has_flow=has_flow,
        has_main=has_main,
        dependencies=dependencies,
        tags=tags
    )


def _parse_readme(readme_path: Path) -> tuple:
    """Parse title and description from README.md."""
    try:
        content = readme_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")
        
        title = ""
        description = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
            elif title and line and not line.startswith("#"):
                # First non-header line after title is description
                description = line
                break
        
        return title or readme_path.parent.name, description
        
    except Exception:
        return readme_path.parent.name, ""


def _parse_requirements(req_path: Path) -> List[str]:
    """Parse dependencies from requirements.txt."""
    try:
        content = req_path.read_text(encoding="utf-8", errors="ignore")
        deps = []
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                # Extract package name without version
                pkg = re.split(r"[<>=!~\[]", line)[0].strip()
                if pkg:
                    deps.append(pkg)
        return deps
    except Exception:
        return []


def _infer_tags(name: str, path: Path) -> List[str]:
    """Infer tags from cookbook name and content."""
    tags = []
    
    # Tags from name
    name_lower = name.lower()
    
    tag_keywords = {
        "agent": ["agent"],
        "rag": ["rag", "retrieval"],
        "chat": ["chat", "conversation"],
        "tool": ["tool"],
        "async": ["async", "parallel"],
        "batch": ["batch"],
        "api": ["api", "fastapi", "websocket"],
        "sql": ["sql", "database"],
        "voice": ["voice", "audio"],
        "vision": ["vision", "pdf", "image"],
        "workflow": ["workflow"],
        "thinking": ["thinking", "reasoning"],
        "search": ["search", "crawl"],
        "hitl": ["hitl", "human-in-the-loop"],
        "streaming": ["streaming"],
        "memory": ["memory"],
    }
    
    for tag, keywords in tag_keywords.items():
        if any(kw in name_lower for kw in keywords):
            tags.append(tag)
    
    # Check manifest for explicit tags
    manifest_path = path / "cookbook_manifest.yaml"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
                if manifest and "tags" in manifest:
                    tags.extend(manifest["tags"])
        except Exception:
            pass
    
    return list(set(tags))


def load_cookbook_adapter(
    cookbook_info: CookbookInfo,
    auto_generate: bool = True
) -> Optional[CookbookAdapter]:
    """
    Load an adapter for a cookbook.
    
    Tries in order:
    1. Explicit adapter.py file
    2. Generate from cookbook_manifest.yaml
    3. Auto-generate from code analysis (if auto_generate=True)
    """
    # Try explicit adapter
    if cookbook_info.has_adapter:
        adapter = _load_explicit_adapter(cookbook_info)
        if adapter:
            return adapter
    
    # Try manifest-based adapter
    if cookbook_info.has_manifest:
        adapter = _load_manifest_adapter(cookbook_info)
        if adapter:
            return adapter
    
    # Try auto-generation
    if auto_generate:
        adapter = _generate_adapter(cookbook_info)
        if adapter:
            return adapter
    
    return None


def _load_explicit_adapter(info: CookbookInfo) -> Optional[CookbookAdapter]:
    """Load adapter from adapter.py file."""
    adapter_path = info.path / "adapter.py"
    
    if not adapter_path.exists():
        return None
    
    try:
        # Add cookbook path to sys.path temporarily
        sys.path.insert(0, str(info.path))
        sys.path.insert(0, str(info.path.parent.parent))  # For pocketflow import
        
        spec = importlib.util.spec_from_file_location(
            f"{info.name}.adapter",
            adapter_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find adapter class
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, CookbookAdapter) and 
                obj is not CookbookAdapter):
                return obj()
        
        # Check for get_adapter function
        if hasattr(module, "get_adapter"):
            return module.get_adapter()
        
        return None
        
    except Exception as e:
        print(f"Warning: Failed to load adapter from {adapter_path}: {e}")
        return None
    finally:
        # Clean up sys.path
        if str(info.path) in sys.path:
            sys.path.remove(str(info.path))


def _load_manifest_adapter(info: CookbookInfo) -> Optional[CookbookAdapter]:
    """Load adapter from cookbook_manifest.yaml."""
    manifest_path = info.path / "cookbook_manifest.yaml"
    
    if not manifest_path.exists():
        return None
    
    try:
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        if not manifest:
            return None
        
        return ManifestAdapter(info, manifest)
        
    except Exception as e:
        print(f"Warning: Failed to load manifest from {manifest_path}: {e}")
        return None


def _generate_adapter(info: CookbookInfo) -> Optional[CookbookAdapter]:
    """Auto-generate an adapter by analyzing cookbook code."""
    try:
        actions = []
        
        # Analyze nodes.py
        if info.has_nodes:
            node_actions = _analyze_nodes_file(info.path / "nodes.py", info)
            actions.extend(node_actions)
        
        # Analyze flow.py
        if info.has_flow:
            flow_actions = _analyze_flow_file(info.path / "flow.py", info)
            actions.extend(flow_actions)
        
        if not actions:
            # Create a generic "run" action
            actions.append(AdapterAction(
                name=f"run_{info.name.replace('-', '_').replace('pocketflow_', '')}",
                description=f"Run the {info.title}",
                parameters={
                    "input": {
                        "type": "str",
                        "description": "Input for the cookbook",
                        "required": True
                    }
                }
            ))
        
        return GeneratedAdapter(info, actions)
        
    except Exception as e:
        print(f"Warning: Failed to generate adapter for {info.name}: {e}")
        return None


def _analyze_nodes_file(nodes_path: Path, info: CookbookInfo) -> List[AdapterAction]:
    """Analyze nodes.py to extract potential actions."""
    actions = []
    
    try:
        content = nodes_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
        
        # Find Node subclasses
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a Node subclass
                is_node = any(
                    (isinstance(base, ast.Name) and base.id in ("Node", "BatchNode", "AsyncNode")) or
                    (isinstance(base, ast.Attribute) and base.attr in ("Node", "BatchNode", "AsyncNode"))
                    for base in node.bases
                )
                
                if is_node:
                    action = _node_class_to_action(node, info)
                    if action:
                        actions.append(action)
        
    except Exception as e:
        pass
    
    return actions


def _node_class_to_action(class_node: ast.ClassDef, info: CookbookInfo) -> Optional[AdapterAction]:
    """Convert a Node class definition to an AdapterAction."""
    class_name = class_node.name
    
    # Skip internal/utility nodes
    skip_patterns = ["Base", "Helper", "Utility", "_"]
    if any(p in class_name for p in skip_patterns):
        return None
    
    # Extract docstring
    description = ""
    if (class_node.body and 
        isinstance(class_node.body[0], ast.Expr) and
        isinstance(class_node.body[0].value, ast.Constant)):
        description = class_node.body[0].value.value
    
    if not description:
        # Generate description from class name
        description = _camel_to_sentence(class_name)
    
    # Create action name
    action_name = _class_name_to_action_name(class_name, info.name)
    
    # Try to infer parameters from prep method
    parameters = _infer_parameters_from_node(class_node)
    
    return AdapterAction(
        name=action_name,
        description=description,
        parameters=parameters
    )


def _analyze_flow_file(flow_path: Path, info: CookbookInfo) -> List[AdapterAction]:
    """Analyze flow.py to extract flow creation functions."""
    actions = []
    
    try:
        content = flow_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
        
        # Find create_*_flow functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if "flow" in node.name.lower() or "create" in node.name.lower():
                    action = _flow_func_to_action(node, info)
                    if action:
                        actions.append(action)
        
    except Exception:
        pass
    
    return actions


def _flow_func_to_action(func_node: ast.FunctionDef, info: CookbookInfo) -> Optional[AdapterAction]:
    """Convert a flow creation function to an AdapterAction."""
    func_name = func_node.name
    
    # Extract docstring
    description = ""
    if (func_node.body and
        isinstance(func_node.body[0], ast.Expr) and
        isinstance(func_node.body[0].value, ast.Constant)):
        description = func_node.body[0].value.value.split("\n")[0]
    
    if not description:
        description = f"Run {_camel_to_sentence(func_name)}"
    
    # Create action name
    action_name = f"run_{info.name.replace('-', '_').replace('pocketflow_', '')}"
    
    return AdapterAction(
        name=action_name,
        description=description,
        parameters={
            "input": {
                "type": "str",
                "description": "Input for the flow",
                "required": False,
                "default": ""
            }
        }
    )


def _camel_to_sentence(name: str) -> str:
    """Convert CamelCase to sentence."""
    # Insert spaces before capitals
    result = re.sub(r"([A-Z])", r" \1", name)
    # Clean up
    result = result.strip().lower()
    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]
    return result


def _class_name_to_action_name(class_name: str, cookbook_name: str) -> str:
    """Convert class name to action name."""
    # Remove common suffixes
    for suffix in ["Node", "Action", "Handler"]:
        if class_name.endswith(suffix):
            class_name = class_name[:-len(suffix)]
    
    # Convert to snake_case
    name = re.sub(r"([A-Z])", r"_\1", class_name).lower().strip("_")
    
    # Add cookbook prefix if needed
    cookbook_prefix = cookbook_name.replace("pocketflow-", "").replace("-", "_")
    
    return f"{cookbook_prefix}_{name}"


def _infer_parameters_from_node(class_node: ast.ClassDef) -> Dict[str, Dict[str, Any]]:
    """Try to infer parameters from a Node class's prep method."""
    params = {}
    
    # Look for prep method
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "prep":
            # Look for shared dictionary accesses
            for node in ast.walk(item):
                if isinstance(node, ast.Subscript):
                    if (isinstance(node.value, ast.Name) and 
                        node.value.id == "shared" and
                        isinstance(node.slice, ast.Constant)):
                        key = node.slice.value
                        if isinstance(key, str) and not key.startswith("_"):
                            params[key] = {
                                "type": "str",
                                "description": f"Value for {key}",
                                "required": True
                            }
            break
    
    return params


class ManifestAdapter(CookbookAdapter):
    """Adapter generated from cookbook_manifest.yaml."""
    
    def __init__(self, info: CookbookInfo, manifest: Dict[str, Any]):
        super().__init__()
        self._info = info
        self._manifest = manifest
        self._actions_list = self._parse_actions()
    
    @property
    def name(self) -> str:
        return self._info.name
    
    @property
    def description(self) -> str:
        return self._manifest.get("description", self._info.description)
    
    @property
    def version(self) -> str:
        return self._manifest.get("version", "1.0.0")
    
    @property
    def tags(self) -> List[str]:
        return self._manifest.get("tags", self._info.tags)
    
    @property
    def dependencies(self) -> List[str]:
        return self._manifest.get("dependencies", self._info.dependencies)
    
    @property
    def actions(self) -> List[AdapterAction]:
        return self._actions_list
    
    def _parse_actions(self) -> List[AdapterAction]:
        """Parse actions from manifest."""
        actions = []
        
        for action_def in self._manifest.get("actions", []):
            action = AdapterAction(
                name=action_def.get("name", "unknown"),
                description=action_def.get("description", ""),
                parameters=action_def.get("parameters", {})
            )
            actions.append(action)
        
        return actions
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action via the cookbook's code."""
        try:
            # Import and run the cookbook
            return self._run_cookbook(action_name, params, shared)
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {e}"
            }
    
    def _run_cookbook(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the cookbook's flow or function."""
        # Add paths
        sys.path.insert(0, str(self._info.path))
        sys.path.insert(0, str(self._info.path.parent.parent))
        
        try:
            # Try to import and run flow
            if self._info.has_flow:
                flow_module = importlib.import_module("flow")
                
                # Find flow creation function
                for name in dir(flow_module):
                    if "flow" in name.lower() and callable(getattr(flow_module, name)):
                        create_flow = getattr(flow_module, name)
                        flow = create_flow()
                        
                        # Merge params into shared
                        flow_shared = {**shared, **params}
                        flow.run(flow_shared)
                        
                        # Extract result
                        result = flow_shared.get("result", flow_shared.get("answer", flow_shared.get("output", "Complete")))
                        
                        return {
                            "success": True,
                            "result": result,
                            "context_update": f"Ran {self.name}: {str(result)[:200]}"
                        }
            
            return {
                "success": False,
                "error": "No runnable flow found"
            }
            
        finally:
            if str(self._info.path) in sys.path:
                sys.path.remove(str(self._info.path))


class GeneratedAdapter(CookbookAdapter):
    """Auto-generated adapter from code analysis."""
    
    def __init__(self, info: CookbookInfo, actions: List[AdapterAction]):
        super().__init__()
        self._info = info
        self._actions_list = actions
    
    @property
    def name(self) -> str:
        return self._info.name
    
    @property
    def description(self) -> str:
        return self._info.description or f"Auto-generated adapter for {self._info.title}"
    
    @property
    def tags(self) -> List[str]:
        return self._info.tags
    
    @property
    def dependencies(self) -> List[str]:
        return self._info.dependencies
    
    @property
    def actions(self) -> List[AdapterAction]:
        return self._actions_list
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute by running the cookbook's flow."""
        try:
            return self._run_cookbook(params, shared)
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {type(e).__name__}: {e}"
            }
    
    def _run_cookbook(
        self,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the cookbook."""
        # Add paths
        sys.path.insert(0, str(self._info.path))
        sys.path.insert(0, str(self._info.path.parent.parent))
        
        try:
            # Try flow.py first
            if self._info.has_flow:
                try:
                    flow_module = importlib.import_module("flow")
                    
                    for name in dir(flow_module):
                        obj = getattr(flow_module, name)
                        if callable(obj) and ("flow" in name.lower() or "create" in name.lower()):
                            flow = obj()
                            flow_shared = {**shared, **params}
                            flow.run(flow_shared)
                            
                            result = self._extract_result(flow_shared)
                            return {
                                "success": True,
                                "result": result,
                                "context_update": f"{self.name}: {str(result)[:300]}"
                            }
                except Exception as e:
                    pass
            
            # Try main.py
            if self._info.has_main:
                try:
                    main_module = importlib.import_module("main")
                    
                    if hasattr(main_module, "main"):
                        # Can't easily capture main() output
                        pass
                except Exception:
                    pass
            
            return {
                "success": False,
                "error": "Could not find runnable code"
            }
            
        finally:
            # Clean up imports
            modules_to_remove = [m for m in sys.modules if m.startswith(("flow", "nodes", "main", "utils"))]
            for m in modules_to_remove:
                try:
                    del sys.modules[m]
                except:
                    pass
            
            if str(self._info.path) in sys.path:
                sys.path.remove(str(self._info.path))
    
    def _extract_result(self, shared: Dict[str, Any]) -> Any:
        """Extract result from shared state."""
        # Common result keys
        for key in ["result", "answer", "output", "response", "final_answer", "generated_answer"]:
            if key in shared:
                return shared[key]
        
        # Return entire shared minus internal keys
        return {k: v for k, v in shared.items() if not k.startswith("_")}
