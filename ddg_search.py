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
            
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search DuckDuckGo and return results
        
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

async def main():
    """Example usage of DuckDuckGoSearch"""
    async with DuckDuckGoSearch() as ddg:
        # Search for something
        query = "fortnite"
        results = await ddg.search(query, num_results=5)
        
        # Print the results
        print(f"Search results for '{query}':")
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Title: {result['title']}")
            print(f"URL: {result['link']}")
            print(f"Snippet: {result['snippet']}")
            print(f"Display Link: {result['displayLink']}")

if __name__ == "__main__":
    asyncio.run(main()) 