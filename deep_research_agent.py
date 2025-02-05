import os
import json
from typing import List, Dict, Optional
import google.generativeai as genai
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time

class DeepResearchAgent:
    def __init__(self):
        load_dotenv()
        
        # Initialize Google Gemini
        genai.configure(api_key=os.getenv('GOOGLE_AI_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize Google Custom Search
        self.search_engine = build(
            "customsearch", "v1", 
            developerKey=os.getenv('GOOGLE_SEARCH_KEY')
        ).cse()
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

    def generate_subqueries(self, main_query: str) -> List[str]:
        """Generate sub-queries using AI to explore different aspects of the main query."""
        prompt = f"""Given the user's query: '{main_query}', generate a list of 10-20 specific 
        and diverse sub-queries that would be helpful in thoroughly researching this topic. 
        These sub-queries should explore different facets of the main query, aiming to cover 
        a wide range of relevant information. Focus on questions that are likely to have 
        answers available on the web, and that require some research.
        Format each sub-query on a new line, starting with a number and a period."""
        
        try:
            response = self.model.generate_content(prompt)
            # Parse the response to extract sub-queries (assuming one per line)
            subqueries = [q.strip() for q in response.text.split('\n') if q.strip() and any(c.isdigit() for c in q)]
            return subqueries[:20]  # Limit to max 20 queries
        except Exception as e:
            print(f"Error generating sub-queries: {e}")
            return [main_query]  # Return original query as fallback

    def web_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform web search for a given query and return results."""
        try:
            results = self.search_engine.list(
                q=query,
                cx=self.search_engine_id,
                num=num_results
            ).execute()
            
            search_results = []
            for item in results.get('items', []):
                search_results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'domain': item.get('displayLink', '')
                })
            return search_results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def rank_results(self, main_query: str, search_results: List[Dict]) -> List[Dict]:
        """Rank search results based on relevance using AI."""
        if not search_results:
            return []

        prompt = f"""For the query: '{main_query}', rank these search results based on relevance.
        For each result, respond with ONLY a number between 0 and 1 (where 1 is highly relevant),
        one score per line. Example format:
        0.95
        0.82
        0.75
        
        Here are the results to rank:
        """ + "\n\n".join([
            f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}"
            for result in search_results
        ])

        try:
            response = self.model.generate_content(prompt)
            scores = []
            
            # Parse scores line by line
            for line in response.text.strip().split('\n'):
                try:
                    score = float(line.strip())
                    if 0 <= score <= 1:
                        scores.append(score)
                except ValueError:
                    continue
            
            # If we don't get enough scores, pad with default values
            while len(scores) < len(search_results):
                scores.append(0.5)
            
            # Truncate extra scores if necessary
            scores = scores[:len(search_results)]
            
            # Add scores to results
            for result, score in zip(search_results, scores):
                result['relevance_score'] = score
                
            return sorted(search_results, key=lambda x: x.get('relevance_score', 0), reverse=True)
        except Exception as e:
            print(f"Error ranking results: {e}")
            # Return original results with default ranking if there's an error
            for i, result in enumerate(search_results):
                result['relevance_score'] = 0.5
            return search_results

    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'aside']):
                element.decompose()
            
            # Extract text from paragraphs and headings
            content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            content = ' '.join(elem.get_text().strip() for elem in content_elements)
            return content
        except Exception as e:
            print(f"Extraction error for {url}: {e}")
            return ""

    def generate_report(self, main_query: str, research_data: Dict) -> str:
        """Generate a comprehensive report from the gathered research data."""
        prompt = f"""Generate a comprehensive research report on: '{main_query}'
        Using the following information:
        {json.dumps(research_data, indent=2)}
        
        Organize the report with clear sections, including:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Sources and Citations
        5. Research Methodology
        
        Make it detailed yet easy to understand.
        Format the report in Markdown."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating report: {e}")
            return f"Error generating report. Please try again.\nError: {str(e)}"

    def research(self, query: str) -> str:
        """Main research function that coordinates the entire research process."""
        print(f"Starting research on: {query}")
        
        # Generate sub-queries
        subqueries = self.generate_subqueries(query)
        print(f"Generated {len(subqueries)} sub-queries")
        
        research_data = {
            'main_query': query,
            'subqueries': [],
            'sources': []
        }
        
        # Process each sub-query
        for subquery in subqueries:
            print(f"Processing sub-query: {subquery}")
            
            # Search and rank results
            results = self.web_search(subquery)
            if results:
                ranked_results = self.rank_results(query, results)
                
                # Deep dive into top results
                top_results = ranked_results[:3]
                subquery_data = {
                    'query': subquery,
                    'findings': []
                }
                
                for result in top_results:
                    content = self.extract_content(result['url'])
                    if content:
                        subquery_data['findings'].append({
                            'source': result['url'],
                            'content': content[:1000],  # Limit content length
                            'relevance_score': result.get('relevance_score', 0.5)
                        })
                
                if subquery_data['findings']:
                    research_data['subqueries'].append(subquery_data)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        # Generate final report
        print("Generating final report...")
        report = self.generate_report(query, research_data)
        return report

def main():
    # Example usage
    agent = DeepResearchAgent()
    query = input("Enter your research query: ")
    report = agent.research(query)
    print("\nResearch Report:")
    print(report)

if __name__ == "__main__":
    main() 