import asyncio
from typing import List, Dict
from utils.browsing import BrowserManager

class DuckDuckGoSearch:
    """Simple utility class for searching DuckDuckGo"""
    
    def __init__(self):
        """Initialize the search utility"""
        self.browser_manager = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        
    async def initialize(self):
        """Initialize the browser manager"""
        self.browser_manager = BrowserManager()
        await self.browser_manager.initialize()
        return self
        
    async def cleanup(self):
        """Clean up resources"""
        if self.browser_manager:
            await self.browser_manager.cleanup()
            
    async def search_html(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search DuckDuckGo HTML version and return results
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of search results as dictionaries with keys:
            - title: Result title
            - link: Result URL
            - snippet: Result description
            - displayLink: Display URL
        """
        if not self.browser_manager:
            await self.initialize()
            
        return await self.browser_manager.search_duckduckgo(query, num_results)
    
    async def search_regular(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search regular DuckDuckGo (not HTML-only version) and return results
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of search results as dictionaries with keys:
            - title: Result title
            - link: Result URL
            - snippet: Result description
            - displayLink: Display URL
        """
        if not self.browser_manager:
            await self.initialize()
            
        return await self.browser_manager.search_duckduckgo_regular(query, num_results)
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search DuckDuckGo with fallback mechanism:
        1. Try regular DuckDuckGo first
        2. If that fails, try HTML DuckDuckGo
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of search results as dictionaries
        """
        if not self.browser_manager:
            await self.initialize()
        
        # Try regular DuckDuckGo first
        try:
            results = await self.search_regular(query, num_results)
            if results and len(results) > 0:
                return results
        except Exception as e:
            print(f"Regular DuckDuckGo search failed: {str(e)}")
        
        # Fall back to HTML DuckDuckGo
        try:
            return await self.search_html(query, num_results)
        except Exception as e:
            print(f"HTML DuckDuckGo search failed: {str(e)}")
            return []  # Return empty list if both methods fail

async def main():
    """Example usage of DuckDuckGoSearch"""
    async with DuckDuckGoSearch() as ddg:
        # Search query
        query = "weather"
        
        # Try regular DuckDuckGo
        print(f"\n--- Regular DuckDuckGo Search for '{query}' ---")
        regular_results = await ddg.search_regular(query, num_results=3)
        
        for i, result in enumerate(regular_results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['link']}")
            print(f"Snippet: {result['snippet']}")
            
        # Try HTML DuckDuckGo
        print(f"\n--- HTML DuckDuckGo Search for '{query}' ---")
        html_results = await ddg.search_html(query, num_results=3)
        
        for i, result in enumerate(html_results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['link']}")
            print(f"Snippet: {result['snippet']}")
            
        # Try combined search with fallback
        print(f"\n--- Combined Search with Fallback for '{query}' ---")
        combined_results = await ddg.search(query, num_results=3)
        
        for i, result in enumerate(combined_results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['link']}")
            print(f"Snippet: {result['snippet']}")

if __name__ == "__main__":
    asyncio.run(main()) 