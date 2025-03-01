from googlesearch import search
from dataclasses import dataclass
from typing import List, Optional, Generator
from utils.browsing import BrowserManager

@dataclass
class SearchResult:
    """Class to mimic Google CSE result structure"""
    url: str
    title: str
    description: str
    
    def to_cse_format(self) -> dict:
        """Convert to Google CSE format"""
        return {
            'link': self.url,
            'title': self.title,
            'snippet': self.description,
            'displayLink': self.url.split('/')[2]  # Extract domain from URL
        }

class MockSearchEngine:
    """Mock class to mimic Google CSE interface"""
    def __init__(self):
        self.browser_manager = None
        
    async def initialize_browser(self):
        """Initialize browser manager for fallback search"""
        if not self.browser_manager:
            self.browser_manager = BrowserManager()
            await self.browser_manager.initialize()
            
    async def cleanup(self):
        """Clean up browser resources"""
        if self.browser_manager:
            await self.browser_manager.cleanup()
        
    async def list(self, q: str, cx: str, num: int = 15) -> 'MockSearchResponse':
        """Mimic CSE list method with DuckDuckGo fallback"""
        try:
            # Try Google search first
            results_generator = search(q, advanced=True, num_results=8, safe="none")
            results = []
            
            # Safely iterate through generator
            for result in results_generator:
                try:
                    # Safely get properties with fallbacks
                    url = getattr(result, 'url', None)
                    if not url:  # Skip if no URL
                        continue
                        
                    title = getattr(result, 'title', None)
                    description = getattr(result, 'description', None)
                    
                    search_result = SearchResult(
                        url=url,
                        title=title if title else url,  # Fallback to URL if no title
                        description=description if description else ''  # Empty string if no description
                    )
                    results.append(search_result)
                    
                    # Break if we have enough results
                    if len(results) >= num:
                        break
                        
                except Exception as e:
                    print(f"Error processing search result: {e}")
                    continue
            
            # Convert to CSE format
            items = [
                result.to_cse_format() 
                for result in results
            ]
            
            return MockSearchResponse(items)
            
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's the Google "sorry" error
            if "sorry" in error_str:
                print("Google search failed with 'sorry' error, falling back to DuckDuckGo")
                try:
                    # Initialize browser if needed
                    await self.initialize_browser()
                    
                    # Use DuckDuckGo as fallback
                    ddg_results = await self.browser_manager.search_duckduckgo(q, num_results=num)
                    return MockSearchResponse(ddg_results)
                    
                except Exception as ddg_error:
                    print(f"DuckDuckGo fallback error: {str(ddg_error)}")
                    return MockSearchResponse([])  # Return empty response if both fail
            else:
                print(f"Search error: {str(e)}")
                return MockSearchResponse([])  # Return empty response on other errors

class MockSearchResponse:
    """Mock class to mimic CSE response"""
    def __init__(self, items: List[dict]):
        self.items = items
    
    def execute(self, http=None) -> dict:
        """Mimic CSE execute method"""
        return {
            'items': self.items
        }

class MockService:
    """Mock service class to handle the cse method properly"""
    def __init__(self):
        self._search_engine = MockSearchEngine()
    
    def cse(self):
        """Return the search engine instance"""
        return self._search_engine
        
    async def cleanup(self):
        """Clean up resources"""
        await self._search_engine.cleanup()

def build(service: str, version: str, developerKey: str = None):
    """Mock function to replace Google API build"""
    return MockService()

async def main():
    """Test the search functionality"""
    service = build('customsearch', 'v1')
    try:
        # Test with a query that might trigger the "sorry" error
        query = "test search"
        search_engine = service.cse()
        response = await search_engine.list(q=query, cx='', num=5)
        results = response.execute()
        
        print(f"Search results for '{query}':")
        for i, item in enumerate(results['items'], 1):
            print(f"\n--- Result {i} ---")
            print(f"Title: {item['title']}")
            print(f"URL: {item['link']}")
            print(f"Snippet: {item['snippet']}")
            print(f"Display Link: {item['displayLink']}")
            
    finally:
        # Clean up resources
        await service.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

