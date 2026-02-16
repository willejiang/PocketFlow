"""
Adapter for pocketflow-tool-crawler cookbook.

Provides web crawling capabilities.
"""

import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class CrawlerAdapter(CookbookAdapter):
    """Adapter for the web crawler cookbook."""
    
    def __init__(self):
        super().__init__()
        self._crawled = {}
    
    @property
    def name(self) -> str:
        return "pocketflow-tool-crawler"
    
    @property
    def description(self) -> str:
        return "Crawl websites and extract content with analysis"
    
    @property
    def tags(self) -> List[str]:
        return ["search", "tool", "crawler"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "requests", "beautifulsoup4"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="crawl_url",
                description="Fetch and extract content from a URL",
                parameters={
                    "url": {"type": "str", "description": "URL to crawl", "required": True}
                }
            ),
            AdapterAction(
                name="crawl_site",
                description="Crawl multiple pages from a website",
                parameters={
                    "base_url": {"type": "str", "description": "Base URL to start crawling", "required": True},
                    "max_pages": {"type": "int", "description": "Maximum pages to crawl", "required": False, "default": 5}
                }
            )
        ]
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action_name == "crawl_url":
            return self._crawl_url(params)
        elif action_name == "crawl_site":
            return self._crawl_site(params)
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _crawl_url(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crawl a single URL."""
        url = params.get("url", "")
        
        if not url:
            return {"success": False, "error": "URL required"}
        
        try:
            # Try using the cookbook's crawler
            from tools.crawler import WebCrawler
            
            crawler = WebCrawler(url, max_pages=1)
            results = crawler.crawl()
            
            if results:
                page = results[0]
                content = page.get("content", "")[:2000]
                
                return {
                    "success": True,
                    "result": {
                        "url": url,
                        "title": page.get("title", ""),
                        "content": content
                    },
                    "context_update": f"Crawled {url}:\n{content}"
                }
            else:
                return {"success": False, "error": "No content retrieved"}
                
        except ImportError:
            # Fallback: simple fetch
            return self._simple_fetch(url)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _simple_fetch(self, url: str) -> Dict[str, Any]:
        """Simple URL fetch fallback."""
        import urllib.request
        import re
        
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PocketFlowBot/1.0)"}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode("utf-8", errors="ignore")
            
            # Extract title
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Untitled"
            
            # Extract text
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()[:2000]
            
            self._crawled[url] = {"title": title, "content": text}
            
            return {
                "success": True,
                "result": {"url": url, "title": title, "content": text},
                "context_update": f"Fetched {url}:\nTitle: {title}\n{text[:500]}..."
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _crawl_site(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crawl multiple pages."""
        base_url = params.get("base_url", "")
        max_pages = params.get("max_pages", 5)
        
        if not base_url:
            return {"success": False, "error": "base_url required"}
        
        try:
            from tools.crawler import WebCrawler
            
            crawler = WebCrawler(base_url, max_pages=max_pages)
            results = crawler.crawl()
            
            context = f"Crawled {len(results)} pages from {base_url}:\n"
            for page in results[:5]:
                title = page.get("title", "")[:50]
                context += f"- {title}\n"
            
            return {
                "success": True,
                "result": results,
                "context_update": context
            }
            
        except ImportError:
            # Just crawl the main URL
            return self._simple_fetch(base_url)
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_adapter() -> CookbookAdapter:
    return CrawlerAdapter()
