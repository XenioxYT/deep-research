from googlesearch import search
from dataclasses import dataclass
from typing import List, Optional, Generator
from utils.browsing import BrowserManager
import asyncio

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
        
    def list(self, q: str, cx: str, num: int = 15) -> 'MockSearchResponse':
        """Mimic CSE list method with Google search only (sync method)"""
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
            # For the Google "sorry" error or any other error, just return empty results
            # We can't do async operations in this sync method
            print(f"Google search error: {str(e)}")
            return MockSearchResponse([])  # Return empty response on errors

    async def async_search_fallback(self, q: str, num: int = 15):
        """
        Async method for DuckDuckGo fallback search (HTML version only)
        This is kept for backward compatibility
        """
        try:
            # Initialize browser if needed
            await self.initialize_browser()
            
            # Use DuckDuckGo HTML as fallback
            ddg_results = await self.browser_manager.search_duckduckgo(q, num_results=num)
            return MockSearchResponse(ddg_results)
            
        except Exception as ddg_error:
            print(f"DuckDuckGo HTML fallback error: {str(ddg_error)}")
            return MockSearchResponse([])  # Return empty response if both fail
            
    async def async_search(self, q: str, num: int = 15):
        """
        Async method with full fallback chain:
        1. Try Google search first
        2. If that fails, try regular DuckDuckGo
        3. If that fails, try HTML DuckDuckGo
        """
        # Try Google search first
        try:
            # Use the synchronous Google search method
            google_response = self.list(q=q, cx='', num=num)
            google_results = google_response.execute()
            
            # Check if we got any results
            if google_results.get('items', []):
                print(f"Successfully retrieved {len(google_results['items'])} results from Google")
                return MockSearchResponse(google_results['items'])
            else:
                print("Google search returned no results, trying DuckDuckGo...")
        except Exception as google_error:
            print(f"Google search error: {str(google_error)}")
        
        # Initialize browser if needed for DuckDuckGo searches
        await self.initialize_browser()
        
        # Try regular DuckDuckGo next
        try:
            regular_ddg_results = await self.browser_manager.search_duckduckgo_regular(q, num_results=num)
            if regular_ddg_results:
                print(f"Successfully retrieved {len(regular_ddg_results)} results from regular DuckDuckGo")
                return MockSearchResponse(regular_ddg_results)
            else:
                print("Regular DuckDuckGo search returned no results, trying HTML DuckDuckGo...")
        except Exception as regular_ddg_error:
            print(f"Regular DuckDuckGo search error: {str(regular_ddg_error)}")
        
        # Finally, try HTML DuckDuckGo as last resort
        try:
            html_ddg_results = await self.browser_manager.search_duckduckgo(q, num_results=num)
            if html_ddg_results:
                print(f"Successfully retrieved {len(html_ddg_results)} results from HTML DuckDuckGo")
                return MockSearchResponse(html_ddg_results)
            else:
                print("HTML DuckDuckGo search returned no results")
        except Exception as html_ddg_error:
            print(f"HTML DuckDuckGo search error: {str(html_ddg_error)}")
        
        # If all methods fail, return empty results
        print("All search methods failed, returning empty results")
        return MockSearchResponse([])

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
        # Test with a query
        query = "test search"
        search_engine = service.cse()
        
        # Use synchronous search method (Google only)
        print("\n=== Testing Google Search ===")
        response = search_engine.list(q=query, cx='', num=5)
        results = response.execute()
        
        if results.get('items'):
            print(f"Google search results for '{query}':")
            for i, item in enumerate(results['items'], 1):
                print(f"\n--- Result {i} ---")
                print(f"Title: {item['title']}")
                print(f"URL: {item['link']}")
                print(f"Snippet: {item['snippet']}")
                print(f"Display Link: {item['displayLink']}")
        else:
            print(f"No Google search results found for '{query}'")
            
        # Test HTML DuckDuckGo fallback
        print("\n=== Testing HTML DuckDuckGo Fallback ===")
        html_ddg_response = await search_engine.async_search_fallback(query, num=3)
        html_ddg_results = html_ddg_response.execute()
        
        if html_ddg_results.get('items'):
            print(f"HTML DuckDuckGo fallback results for '{query}':")
            for i, item in enumerate(html_ddg_results['items'], 1):
                print(f"\n--- HTML DuckDuckGo Result {i} ---")
                print(f"Title: {item['title']}")
                print(f"URL: {item['link']}")
                print(f"Snippet: {item['snippet']}")
                print(f"Display Link: {item.get('displayLink', '')}")
        else:
            print(f"No HTML DuckDuckGo fallback results found for '{query}'")
            
        # Test full fallback chain
        print("\n=== Testing Full Fallback Chain ===")
        print("This will try: Google → Regular DuckDuckGo → HTML DuckDuckGo")
        fallback_response = await search_engine.async_search(query, num=3)
        fallback_results = fallback_response.execute()
        
        if fallback_results.get('items'):
            print(f"Full fallback chain results for '{query}':")
            for i, item in enumerate(fallback_results['items'], 1):
                print(f"\n--- Fallback Chain Result {i} ---")
                print(f"Title: {item['title']}")
                print(f"URL: {item['link']}")
                print(f"Snippet: {item['snippet']}")
                print(f"Display Link: {item.get('displayLink', '')}")
        else:
            print(f"No results found in the entire fallback chain for '{query}'")
            
    finally:
        # Clean up resources
        await service.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

