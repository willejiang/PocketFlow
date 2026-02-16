"""
Adapter for pocketflow-text2sql cookbook.

Provides natural language to SQL capabilities.
"""

import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class Text2SQLAdapter(CookbookAdapter):
    """Adapter for the text2sql cookbook."""
    
    def __init__(self):
        super().__init__()
        self._connections = {}
        self._schemas = {}
    
    @property
    def name(self) -> str:
        return "pocketflow-text2sql"
    
    @property
    def description(self) -> str:
        return "Convert natural language to SQL and query databases"
    
    @property
    def tags(self) -> List[str]:
        return ["sql", "database", "query"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "pyyaml"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="sql_connect",
                description="Connect to a SQLite database",
                parameters={
                    "db_path": {
                        "type": "str",
                        "description": "Path to SQLite database file",
                        "required": True
                    },
                    "alias": {
                        "type": "str",
                        "description": "Alias for this connection",
                        "required": False,
                        "default": "default"
                    }
                }
            ),
            AdapterAction(
                name="sql_query",
                description="Query the database using natural language",
                parameters={
                    "query": {
                        "type": "str",
                        "description": "Natural language query",
                        "required": True
                    },
                    "alias": {
                        "type": "str",
                        "description": "Database alias",
                        "required": False,
                        "default": "default"
                    }
                }
            ),
            AdapterAction(
                name="sql_execute",
                description="Execute raw SQL query",
                parameters={
                    "sql": {
                        "type": "str",
                        "description": "SQL query to execute",
                        "required": True
                    },
                    "alias": {
                        "type": "str",
                        "description": "Database alias",
                        "required": False,
                        "default": "default"
                    }
                }
            )
        ]
    
    def cleanup(self, shared: Dict[str, Any]) -> None:
        """Close all connections."""
        for conn in self._connections.values():
            try:
                conn.close()
            except:
                pass
        self._connections.clear()
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action_name == "sql_connect":
            return self._connect(params)
        elif action_name == "sql_query":
            return self._natural_query(params)
        elif action_name == "sql_execute":
            return self._execute_sql(params)
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _connect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to database."""
        import sqlite3
        
        db_path = params.get("db_path", "")
        alias = params.get("alias", "default")
        
        if not db_path:
            return {"success": False, "error": "db_path is required"}
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # Close existing
            if alias in self._connections:
                try:
                    self._connections[alias].close()
                except:
                    pass
            
            self._connections[alias] = conn
            
            # Get schema
            schema = self._get_schema(conn)
            self._schemas[alias] = schema
            
            return {
                "success": True,
                "result": f"Connected to {db_path}",
                "context_update": f"Database connected ({alias}):\n{schema}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_schema(self, conn) -> str:
        """Get database schema."""
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        schema_lines = []
        for (table_name,) in tables:
            schema_lines.append(f"Table: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name})")
            for col in cursor.fetchall():
                schema_lines.append(f"  - {col[1]} ({col[2]})")
            schema_lines.append("")
        
        return "\n".join(schema_lines)
    
    def _natural_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert natural language to SQL and execute."""
        query = params.get("query", "")
        alias = params.get("alias", "default")
        
        if not query:
            return {"success": False, "error": "Query is required"}
        
        if alias not in self._connections:
            return {"success": False, "error": f"Database '{alias}' not connected"}
        
        schema = self._schemas.get(alias, "")
        
        try:
            try:
                from utils.call_llm import call_llm
            except ImportError:
                from utils import call_llm
            
            import yaml
            
            prompt = f"""Given this SQLite schema:
{schema}

Convert this natural language query to SQL:
"{query}"

Respond with YAML:
```yaml
sql: |
  SELECT ...
```"""
            
            response = call_llm(prompt)
            yaml_str = response.split("```yaml")[1].split("```")[0].strip()
            result = yaml.safe_load(yaml_str)
            sql = result["sql"].strip().rstrip(";")
            
            # Execute
            return self._execute_sql({"sql": sql, "alias": alias})
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_sql(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL query."""
        sql = params.get("sql", "")
        alias = params.get("alias", "default")
        
        if not sql:
            return {"success": False, "error": "SQL is required"}
        
        if alias not in self._connections:
            return {"success": False, "error": f"Database '{alias}' not connected"}
        
        conn = self._connections[alias]
        
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            
            is_select = sql.strip().upper().startswith(("SELECT", "WITH"))
            
            if is_select:
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                results = [dict(row) for row in rows]
                
                # Format for context
                context = f"SQL: {sql}\n\nResults ({len(results)} rows):\n"
                if columns:
                    context += " | ".join(columns) + "\n"
                    context += "-" * 40 + "\n"
                for row in results[:10]:
                    context += " | ".join(str(v) for v in row.values()) + "\n"
                
                return {
                    "success": True,
                    "result": {"columns": columns, "rows": results},
                    "context_update": context
                }
            else:
                conn.commit()
                return {
                    "success": True,
                    "result": f"Affected rows: {cursor.rowcount}",
                    "context_update": f"SQL executed: {sql}\nAffected: {cursor.rowcount}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_adapter() -> CookbookAdapter:
    return Text2SQLAdapter()
