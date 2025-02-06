from googlesearch import search
from dataclasses import dataclass
from typing import List, Optional, Generator

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
        pass
        
    def list(self, q: str, cx: str, num: int = 12) -> 'MockSearchResponse':
        """Mimic CSE list method"""
        try:
            # Get raw results and handle empty case
            raw_results = list(search(q, advanced=True, num_results=num, safe="none"))
            if not raw_results:
                return MockSearchResponse([])
            
            # Convert raw results to our SearchResult objects with safer property access
            results = []
            for result in raw_results:
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
            print(f"Search error: {e}")
            return MockSearchResponse([])  # Return empty response on error

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

def build(service: str, version: str, developerKey: str = None):
    """Mock function to replace Google API build"""
    return MockService()

