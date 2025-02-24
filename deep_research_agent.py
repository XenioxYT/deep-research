import os
import json
from typing import List, Dict, Optional, Set, Tuple
import google.generativeai as genai
from test_search import build
from dotenv import load_dotenv
import time
from collections import defaultdict
import logging
import colorlog
from datetime import datetime
import re
import asyncio
import httplib2
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from vertexai.preview import tokenization
from utils.browsing import BrowserManager

class DeepResearchAgent:
    def __init__(self):
        """Regular initialization of non-async components."""
        load_dotenv()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Google Gemini with context
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.approximate_location = "UK"
        system_context = f"Current date: {self.current_date}. Location: {self.approximate_location}. "
        system_context += "Only use the information from web searches, not your training data. "
        system_context += "For queries about current/latest things, use generic search terms without specific versions/numbers."
        
        genai.configure(api_key=os.getenv('GOOGLE_AI_KEY'))
        
        # Define common safety settings for all models
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Initialize different models for different tasks with safety settings
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-lite-preview-02-05',
            safety_settings=self.safety_settings
        )  # Default model for general tasks
        
        self.ranking_model = genai.GenerativeModel(
            'gemini-2.0-flash-lite-preview-02-05',
            safety_settings=self.safety_settings
        )  # Specific model for ranking
        
        self.analysis_model = genai.GenerativeModel(
            'gemini-2.0-flash-lite-preview-02-05',
            safety_settings=self.safety_settings
        )  # Model for analysis
        
        self.report_model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            safety_settings=self.safety_settings
        )  # Model for final report generation
        
        # Initialize Google Custom Search
        self.search_engine = build(
            "customsearch", "v1", 
            developerKey=os.getenv('GOOGLE_SEARCH_KEY')
        ).cse()
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        # Research state
        self.previous_queries = set()  # Changed to set for uniqueness
        self.all_results = {}
        self.high_ranking_urls = {}  # Track URLs with score > 0.6
        self.blacklisted_urls = set()
        self.scraped_urls = set()  # Track already scraped URLs
        self.research_iterations = 0
        self.MAX_ITERATIONS = 5
        self.system_context = system_context
        self.total_tokens = 0  # Track total tokens used
        
        # Initialize tokenizer for Gemini model
        self.model_name = "gemini-1.5-flash-002"
        self.tokenizer = tokenization.get_tokenizer_for_model(self.model_name)
        self.token_usage_by_operation = defaultdict(int)
        self.content_tokens = 0  # Track tokens from stored content separately
        
        # Initialize browser manager
        self.browser_manager = None

    async def __aenter__(self):
        """Async initialization when entering context."""
        self.browser_manager = await BrowserManager().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context."""
        if self.browser_manager:
            await self.browser_manager.__aexit__(exc_type, exc_val, exc_tb)

    def setup_logging(self):
        """Setup colorized logging."""
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s:%(reset)s %(message)s',
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING': 'yellow',
                'ERROR':   'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        
        logger = colorlog.getLogger('deep_research')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        self.logger = logger

    def should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped."""
        return (
            url in self.blacklisted_urls or
            any(ext in url for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx'])
        )

    def generate_subqueries(self, main_query: str, research_state: Optional[Dict] = None) -> List[str]:
        """Generate sub-queries using AI to explore different aspects of the main query."""
        self.logger.info("Analyzing query and generating search queries...")
        
        MAX_QUERIES = 5  # Maximum number of queries to return
        
        context = ""
        if research_state and self.previous_queries:
            context = f"""
            Previously used queries: {json.dumps(list(self.previous_queries), indent=2)}
            Current research state: {json.dumps(research_state, indent=2)}
            Based on the above context and gaps in current research, """
        
        prompt = f"""{self.system_context}
        {context}Generate comprehensive search queries to gather maximum relevant information about this query:

        Query: '{main_query}'

        First, determine if this is a SIMPLE query (basic math, unit conversion, single fact lookup) or a COMPLEX query requiring research.

        For COMPLEX queries, generate search queries that:
        1. Cover all temporal aspects:
           - Historical background and development
           - Current state and recent developments
           - Future predictions and trends

        2. Include different information types:
           - Core facts and definitions
           - Statistics and data
           - Expert analysis and opinions
           - Case studies and examples
           - Comparisons and contrasts
           - Problems and solutions
           - Impacts and implications

        3. Use search optimization techniques:
           - Site-specific searches (e.g., site:edu, site:gov)
           - Date-range specific queries when relevant
           - Include synonyms and related terms
           - Combine key concepts in different ways
           - Use both broad and specific queries

        4. Response Format:
        TYPE: [SIMPLE/COMPLEX]
        REASON: [One clear sentence explaining the classification]
        QUERIES:
        [If SIMPLE: Only output the original query
        If COMPLEX: Generate up to {MAX_QUERIES} search queries that:
        - Start each line with a number
        - Ensure broad coverage within the {MAX_QUERIES} query limit]

        Example of a COMPLEX query about "impact of remote work":
        1. "remote work" impact statistics 2023-2024
        2. site:edu research "remote work productivity"
        3. challenges "distributed workforce" solutions
        4. remote work employee mental health studies
        5. "hybrid work model" vs "fully remote" comparison
        6. site:gov telework policy guidelines
        7. "remote work" environmental impact data
        8. distributed teams collaboration best practices
        9. "remote work" industry adoption rates
        10. future "workplace trends" expert predictions"""

        try:
            response = self.model.generate_content(prompt)
            if not response or not response.text:
                self.logger.error("Empty response from AI model")
                return [main_query]
            
            # Parse response
            response_text = response.text.strip()
            query_type = None
            reason = None
            queries_section_started = False
            subqueries = []
            
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('TYPE:'):
                    query_type = line.split(':', 1)[1].strip().upper()
                elif line.startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()
                elif line.startswith('QUERIES:'):
                    queries_section_started = True
                elif queries_section_started:
                    # For SIMPLE queries, just add the main query
                    if query_type == "SIMPLE":
                        subqueries = [main_query]
                        break
                    
                    # For COMPLEX queries, process numbered lines
                    if any(c.isdigit() for c in line):
                        try:
                            # Handle different number formats (1., 1-, 1), etc.
                            query = re.split(r'^\d+[.)-]\s*', line)[-1].strip()
                            
                            # Only validate minimum length, no maximum
                            if query and len(query) >= 3:
                                subqueries.append(query)
                                # Break if we've reached the maximum number of queries
                                if len(subqueries) >= MAX_QUERIES:
                                    break
                        except Exception as e:
                            self.logger.warning(f"Error processing query line '{line}': {e}")
                            continue
            
            # Always include the main query for complex queries if we have room
            if query_type == "COMPLEX" and main_query not in subqueries and len(subqueries) < MAX_QUERIES:
                subqueries.append(main_query)
            
            # Log results
            self.logger.info(f"Query type: {query_type} - {reason}")
            self.logger.info(f"Generated {len(subqueries)} queries:")
            for q in subqueries:
                self.logger.info(f"Query: {q}")
            
            return subqueries if subqueries else [main_query]
            
        except Exception as e:
            self.logger.error(f"Error generating queries: {e}")
            return [main_query]

    async def batch_web_search(self, queries: List[str], num_results: int = 10) -> List[Dict]:
        """Perform multiple web searches in parallel with increased batch size."""
        self.logger.info(f"Batch searching {len(queries)} queries...")
        
        # Increased batch size for better throughput
        batch_size = 10  # Increased from 3
        max_concurrent = 5  # Maximum concurrent API calls
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def search_with_semaphore(query: str) -> List[Dict]:
            """Perform a single search with semaphore control."""
            async with semaphore:
                try:
                    # Add retry logic with exponential backoff
                    max_retries = 3
                    base_delay = 1
                    
                    for attempt in range(max_retries):
                        try:
                            # Create SSL-unverified HTTP client
                            http = httplib2.Http(timeout=30)
                            http.disable_ssl_certificate_validation = True
                            
                            results = await asyncio.to_thread(
                                self.search_engine.list(
                                    q=query,
                                    cx=self.search_engine_id,
                                    num=num_results
                                ).execute,
                                http=http
                            )
                            
                            if not results or 'items' not in results:
                                self.logger.warning(f"No results found for query: {query}")
                                return []
                            
                            search_results = []
                            for item in results.get('items', []):
                                try:
                                    url = item.get('link', '')
                                    if not url or self.should_skip_url(url):
                                        continue
                                        
                                    # Use browser manager's rewrite_url
                                    url = self.browser_manager.rewrite_url(url)
                                    
                                    result = {
                                        'title': item.get('title', ''),
                                        'url': url,
                                        'snippet': item.get('snippet', ''),
                                        'domain': item.get('displayLink', ''),
                                        'source_queries': [query]
                                    }
                                    search_results.append(result)
                                except Exception as item_error:
                                    self.logger.warning(f"Error processing search result: {item_error}")
                                    continue
                            
                            return search_results
                            
                        except Exception as retry_error:
                            if attempt == max_retries - 1:
                                raise
                            delay = base_delay * (2 ** attempt)
                            self.logger.warning(f"Search attempt {attempt + 1} failed: {retry_error}")
                            await asyncio.sleep(delay)
                            
                except Exception as e:
                    self.logger.error(f"Search error for query '{query}': {str(e)}")
                    return []
        
        # Process queries in parallel batches
        all_results = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_tasks = [search_with_semaphore(query) for query in batch_queries]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Process batch results
            for query, query_results in zip(batch_queries, batch_results):
                for result in query_results:
                    url = result['url']
                    if url in self.all_results:
                        self.all_results[url]['source_queries'].append(query)
                    else:
                        self.all_results[url] = result
                        all_results.append(result)
            
            # Small delay between batches to prevent rate limiting
            if i + batch_size < len(queries):
                await asyncio.sleep(0.5)
        
        self.logger.info(f"Found {len(all_results)} unique results across all queries")
        return all_results


    def rank_new_results(self, main_query: str, new_results: List[Dict]) -> List[Dict]:
        """Rank only new search results based on relevance using AI."""
        if not new_results:
            return []

        self.logger.info(f"Ranking {len(new_results)} new URLs")

        prompt = f"""{self.system_context}
        For the query '{main_query}', analyze and rank these search results.
        For each URL, determine:
        1. Relevance score (0-0.99, or 1.0 for perfect matches)
        2. Whether to scrape the content (YES/NO)
        3. Scraping level (LOW/MEDIUM/HIGH) - determines how much content to extract:
           - LOW: 3000 chars - For basic/overview content (default)
           - MEDIUM: 6000 chars - For moderate detail
           - HIGH: 10000 chars - For in-depth analysis
        
        Consider these factors:
        - Content depth and relevance to query
        - Source authority and reliability
        - Need for detailed information
        - If the request is simple (e.g. 2+2), mark none for scraping
        
        Format response EXACTLY as follows, one entry per line:
        [url] | [score] | [YES/NO] | [LOW/MEDIUM/HIGH]
        
        IMPORTANT RULES:
        - All scores must be unique (no ties) and between 0 and 1.0
        - Only give 1.0 for perfect matches
        - Mark YES for scraping only if the content is likely highly relevant
        - Scraping level should match content importance and depth
        - You MUST rank ALL URLs provided
        - Provide scraping decisions for ALL URLs
        
        URLs to analyze:
        """ + "\n".join([
            f"{data['url']}\nTitle: {data['title']}\nSnippet: {data['snippet']}"
            for data in new_results
        ])

        try:
            response = self.ranking_model.generate_content(prompt)
            
            # Parse rankings and verify uniqueness
            rankings = {}
            scrape_decisions = {}
            scrape_count = 0  # Track number of URLs marked for scraping
            
            for line in response.text.strip().split('\n'):
                try:
                    # Split line by | and strip whitespace
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) != 4:  # Updated to expect 4 parts
                        continue
                        
                    url, score_str, scrape_decision, scrape_level = parts
                    score = float(score_str)
                    
                    rankings[url] = score
                    
                    # Only mark for scraping if we haven't hit our limit
                    should_scrape = scrape_decision.upper() == 'YES' and scrape_count < 5
                    if should_scrape:
                        scrape_count += 1
                    
                    # Validate and normalize scraping level
                    scrape_level = scrape_level.upper()
                    if scrape_level not in ['LOW', 'MEDIUM', 'HIGH']:
                        scrape_level = 'MEDIUM'  # Default to LOW if invalid
                    
                    scrape_decisions[url] = {
                        'should_scrape': should_scrape,
                        'scrape_level': scrape_level,
                        'original_decision': scrape_decision.upper() == 'YES'
                    }

                    # Track high-ranking URLs (score > 0.6)
                    if score > 0.6:
                        result = next((r for r in new_results if r['url'] == url), None)
                        if result:
                            self.high_ranking_urls[url] = {
                                'score': score,
                                'title': result['title'],
                                'snippet': result['snippet'],
                                'domain': result['domain'],
                                'source_queries': result['source_queries'],
                                'scrape_decision': scrape_decisions[url]
                            }
                except (ValueError, IndexError):
                    continue
            
            # Update scores and sort results
            ranked_results = []
            for result in new_results:
                if result['url'] in rankings:
                    result['relevance_score'] = rankings[result['url']]
                    result['scrape_decision'] = scrape_decisions[result['url']]
                    ranked_results.append(result)
            
            ranked_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Log summary instead of all URLs
            self.logger.info(
                f"Ranking summary:\n"
                f"Total URLs: {len(ranked_results)}\n"
                f"URLs marked for scraping: {scrape_count}\n"
                f"High-ranking URLs (score > 0.6): {len(self.high_ranking_urls)}"
            )
            
            return ranked_results
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            return new_results

    def get_scrape_limit(self, scrape_level: str) -> int:
        """Get character limit based on scraping level."""
        limits = {
            'LOW': 3000,
            'MEDIUM': 6000,
            'HIGH': 10000
        }
        return limits.get(scrape_level.upper(), 3000)  # Default to LOW if invalid

    def rank_all_results(self, main_query: str) -> List[Dict]:
        """Get all results sorted by their existing ranking scores."""
        if not self.all_results:
            return []

        self.logger.info(f"Getting all {len(self.all_results)} ranked results")
        
        # Simply sort by existing scores
        ranked_results = sorted(
            [r for r in self.all_results.values() if r['url'] not in self.blacklisted_urls],
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        
        return ranked_results

    def analyze_research_state(self, main_query: str, current_data: Dict) -> Tuple[bool, str, List[str], str]:
        """Analyze current research state and decide if more research is needed."""
        self.logger.info("Analyzing research state...")
        
        prompt = f"""{self.system_context}
        Analyze the research on: '{main_query}'
        Current data: {json.dumps(current_data, indent=2)}
        Current iteration: {self.research_iterations}
        
        IMPORTANT DECISION GUIDELINES:
        1. For simple factual queries (e.g. "2+2", "capital of France"), say NO immediately and mark as SIMPLE
        2. For queries needing more depth/verification:
           - On iteration 0-2: Say YES if significant information is missing
           - On iteration 3: Only say YES if crucial information is missing
           - On iteration 4+: Strongly lean towards NO unless absolutely critical information is missing
           - If a section called "Further Research" can be written, continue research.
        3. Consider the query ANSWERED when you have:
           - Sufficient high-quality sources (relevance_score > 0.6)
           - Enough information to provide a comprehensive answer
           - Cross-verified key information from multiple sources
        
        ANALYSIS STEPS:
        1. First, determine if this is a simple factual query requiring no research
        2. If research is needed, assess if current findings sufficiently answer the query
        3. Review all scraped URLs and identify any that are:
           - Not directly relevant to the main query
           - Contain tangential or off-topic information
           - Duplicate or redundant information
           - Low quality or unreliable sources
        4. Only continue research if genuinely valuable information is missing
        5. Generate a custom report structure based on the query type and findings. If the query is simple, the report should be a simple answer to the query. If the query is complex, the report should be a comprehensive report with all the information needed to answer the query.
        6. Mark any unscraped URLs that should be scraped in the next iteration
        
        Format response EXACTLY as follows:
        DECISION: [YES (continue research)/NO (produce final report)]
        TYPE: [SIMPLE/COMPLEX]
        REASON: [One clear sentence explaining the decision, mentioning iteration number if relevant]
        REMOVE_URLS: [List URLs to remove from context, one per line, with brief reason after # symbol]
        BLACKLIST: [List URLs to blacklist, one per line. These URLs will be ignored in future iterations.]
        MISSING: [List missing information aspects, one per line]
        SEARCH_QUERIES: [List complete search queries, one per line, max 7. Search formatting and quotes are allowed. These queries should be specific to the information you are looking for.]
        SCRAPE_NEXT: [List URLs to scrape in next iteration, one per line, in format: URL | LOW/MEDIUM/HIGH]"""
        
        try:
            response = self.analysis_model.generate_content(prompt)
            
            # Parse sections
            decision = False
            query_type = "COMPLEX"  # Default to complex
            blacklist = []
            missing = []
            search_queries = []
            urls_to_scrape_next = {}
            urls_to_remove = {}
            current_section = None
            
            for line in response.text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Handle section headers
                if line.startswith("DECISION:"):
                    decision = "YES" in line.upper()
                    current_section = None
                elif line.startswith("TYPE:"):
                    query_type = line.split(":", 1)[1].strip().upper()
                    current_section = None
                elif line.startswith("REMOVE_URLS:"):
                    current_section = "REMOVE_URLS"
                elif line.startswith("BLACKLIST:"):
                    current_section = "BLACKLIST"
                elif line.startswith("MISSING:"):
                    current_section = "MISSING"
                elif line.startswith("SEARCH_QUERIES:"):
                    current_section = "SEARCH_QUERIES"
                elif line.startswith("SCRAPE_NEXT:"):
                    current_section = "SCRAPE_NEXT"
                elif line.startswith("REPORT_STRUCTURE:"):
                    current_section = "REPORT_STRUCTURE"
                elif line.startswith("REASON:"):  # Add reason to explanation
                    explanation = line.split(":", 1)[1].strip()
                    current_section = None
                else:
                    # Handle section content
                    if current_section == "REMOVE_URLS" and line.startswith('http'):
                        # Parse URL and reason if provided
                        parts = line.split('#', 1)
                        url = parts[0].strip()
                        reason = parts[1].strip() if len(parts) > 1 else "Not relevant to query"
                        urls_to_remove[url] = reason
                    elif current_section == "BLACKLIST" and line.startswith('http'):
                        blacklist.append(line.strip())
                    elif current_section == "MISSING" and line.startswith('-'):
                        missing.append(line[1:].strip())
                    elif current_section == "SEARCH_QUERIES":
                        # Handle multiple query formats
                        if line.startswith('- '):
                            search_queries.append(line[2:].strip())
                        elif line.strip() and not line.startswith(('DECISION:', 'TYPE:', 'BLACKLIST:', 'MISSING:', 'SEARCH_QUERIES:', 'REASON:', 'REPORT_STRUCTURE:', 'SCRAPE_NEXT:', 'REMOVE_URLS:')):
                            # Handle numbered or plain queries
                            clean_query = line.split('. ', 1)[-1] if '. ' in line else line
                            search_queries.append(clean_query.strip())
                    elif current_section == "SCRAPE_NEXT":
                        # Parse URLs marked for scraping in next iteration
                        if '|' in line:
                            url, level = [part.strip() for part in line.split('|')]
                            if level.upper() in ['LOW', 'MEDIUM', 'HIGH']:
                                urls_to_scrape_next[url] = level.upper()
                    elif current_section == "REPORT_STRUCTURE":
                        if report_structure:
                            report_structure += "\n"
                        report_structure += line
            
            # Process URLs to remove
            for url, reason in urls_to_remove.items():
                # Remove from all_results
                if url in self.all_results:
                    del self.all_results[url]
                    self.logger.info(f"Removed URL from context: {url} (Reason: {reason})")
                
                # Remove from high_ranking_urls
                if url in self.high_ranking_urls:
                    del self.high_ranking_urls[url]
                    self.logger.info(f"Removed URL from high-ranking URLs: {url}")
                
                # Add to blacklist to prevent re-scraping
                self.blacklisted_urls.add(url)
                
                # Remove from scraped_urls if present
                if url in self.scraped_urls:
                    self.scraped_urls.remove(url)
            
            # Update blacklist with additional URLs
            self.blacklisted_urls.update(blacklist)
            
            # Update scraping decisions in all_results for next iteration
            for url, level in urls_to_scrape_next.items():
                if url in self.all_results and url not in self.scraped_urls:
                    self.all_results[url]['scrape_decision'] = {
                        'should_scrape': True,
                        'scrape_level': level,
                        'original_decision': True
                    }
                    self.logger.info(f"Marked {url} for {level} scraping in next iteration")
            
            # Log missing information if any
            if missing:
                self.logger.info("Missing information:\n- " + "\n- ".join(missing))
            
            # Prepare explanation (simplified)
            explanation = f"Query type: {query_type}. " + explanation
            if urls_to_remove:
                explanation += f"\nRemoved {len(urls_to_remove)} URLs"
            if search_queries:
                explanation += f"\nGenerated {len(search_queries)} new search queries"
            if urls_to_scrape_next:
                explanation += f"\nMarked {len(urls_to_scrape_next)} URLs for next scraping"
            
            return decision, explanation, search_queries, ""
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return False, str(e), [], ""

    def save_report_streaming(self, query: str, report_text, sources_used: str):
        """Save the report to a markdown file."""
        try:
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Clean and truncate query for filename
            clean_query = self.clean_filename(query)
            
            # Create filename with date
            filename = f"reports/{clean_query}-{self.current_date}.md"
            
            # Write the report content
            with open(filename, 'w', encoding='utf-8') as f:
                try:
                    # Write the report text
                    if report_text:
                        f.write(report_text)
                        self.log_token_usage(report_text, "Report content")
                    
                    # Add the sources section at the end
                    f.write("\n\n")  # Add some spacing
                    f.write(sources_used)
                except Exception as e:
                    self.logger.error(f"Error writing report: {e}")
                    raise  # Re-raise to be caught by outer try-except
            
            self.logger.info(f"Report saved to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            return None

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using Gemini's token counter."""
        if not text:
            return 0
        try:
            # Convert text to string if it's not already
            text = str(text)
            # Count tokens using Gemini's counter
            return self.model.count_tokens(text).total_tokens
        except Exception as e:
            self.logger.warning(f"Token counting error: {e}")
            # Fallback to rough estimation if counter fails
            return len(text) // 4

    def log_token_usage(self, text: str, operation: str):
        """Log token usage for an operation with improved tracking."""
        try:
            tokens = self.count_tokens(text)
            self.total_tokens += tokens
            self.token_usage_by_operation[operation] += tokens

        except Exception as e:
            self.logger.error(f"Error logging token usage: {e}")

    def reset_state(self):
        """Reset all state tracking for a new query."""
        self.previous_queries = set()
        self.all_results = {}
        self.high_ranking_urls = {}
        self.blacklisted_urls = set()
        self.scraped_urls = set()
        self.research_iterations = 0
        self.total_tokens = 0
        self.content_tokens = 0
        self.token_usage_by_operation.clear()
        self.logger.info("Reset research state and token counter")

    def clean_filename(self, query: str, max_length: int = 100) -> str:
        """Clean and truncate query for filename creation."""
        # Remove special characters and convert spaces to hyphens
        clean_query = re.sub(r'[^\w\s-]', '', query).strip().lower()
        clean_query = re.sub(r'[-\s]+', '-', clean_query)
        
        # Truncate if longer than max_length while keeping whole words
        if len(clean_query) > max_length:
            clean_query = clean_query[:max_length].rsplit('-', 1)[0]
        
        # Add current time to the filename
        current_time = datetime.now().strftime("%H-%M-%S")
        return f"{clean_query}-{current_time}"

    async def generate_report(self, main_query: str, research_data: Dict, report_structure: str) -> Tuple[str, str]:
        """Generate a comprehensive report from the gathered research data."""
        max_retries = 3
        base_delay = 2  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Prepare enhanced context with high-ranking URLs and detailed source information
                high_ranking_sources = {}
                for url, data in self.high_ranking_urls.items():
                    # Find the full content from research_data
                    source_content = None
                    for iteration in research_data['iterations']:
                        for finding in iteration['findings']:
                            if finding['source'] == url:
                                source_content = finding['content']
                                break
                        if source_content:
                            break
                    
                    high_ranking_sources[url] = {
                        'title': data['title'],
                        'snippet': data['snippet'],
                        'score': data['score'],
                        'content': source_content,  # Full content if available
                        'domain': data['domain'],
                        'queries_used': data['source_queries']
                    }
                
                # Sort sources by score for easy reference
                sorted_sources = sorted(
                    high_ranking_sources.items(),
                    key=lambda x: x[1]['score'],
                    reverse=True
                )
                
                # Create numbered references for citations
                source_references = {
                    url: f"[{i+1}]" 
                    for i, (url, _) in enumerate(sorted_sources)
                }
                
                # Create source list for the report
                sources_used = "\n\n## Sources Used\n\n"
                for i, (url, data) in enumerate(sorted_sources, 1):
                    sources_used += f"[{i}] {data['title']}\n"
                    sources_used += f"- URL: {url}\n"
                    sources_used += f"- Relevance Score: {data['score']:.2f}\n"
                    sources_used += f"- Domain: {data['domain']}\n\n"
                
                report_context = {
                    'main_query': main_query,
                    'research_data': research_data,
                    'high_ranking_sources': dict(sorted_sources),  # Ordered by score
                    'source_references': source_references,  # For citations
                    'total_sources_analyzed': len(self.all_results),
                    'total_high_ranking_sources': len(self.high_ranking_urls),
                    'research_iterations': self.research_iterations,
                    'total_queries_used': len(self.previous_queries),
                    'queries_by_iteration': [
                        iter_data['queries_used'] 
                        for iter_data in research_data['iterations']
                    ]
                }
                
                prompt = f"""Generate a comprehensive research report on: '{main_query}'

                # For simple queries (mathematical, factual, or definitional):
                - Use # for the main title
                - Use ## for main sections
                - Use ### for subsections if needed
                - Provide a clear, direct answer
                - Include a brief explanation of the concept if relevant
                - Keep additional context minimal and focused

                # For complex queries:
                - Create a title with # heading
                - Use ## for main sections
                - Use ### for subsections
                - Use #### for detailed subsection breakdowns where needed
                - Include comprehensive analysis of all relevant information
                - Address any contradictions or nuances in the sources
                - Provide thorough explanations and context

                # General Guidelines:
                - The report should be detailed and include all relevant information from sources
                - Always use proper heading hierarchy (# → ## → ### → ####)
                - Use **bold** for emphasis on key points
                - Format numbers naturally with proper thousands separators
                - Use [1][2][3] format for references, DO NOT DO [1, 2, 3]
                - Mention when using knowledge beyond the sources and note potential for hallucination
                - Use LaTeX for ALL math expressions by wrapping them in $$. Examples:
                  - For inline math: $$x^2$$ or $$x_2$$
                  - For display math on its own line:
                    $$
                    x^2 + y^2 = z^2
                    $$
                - DO NOT use single $ for math. DO NOT use HTML formatting like <sup> or <sub>
                
                # The report should be comprehensive and thorough:
                - Aim for a length of at least 16,000 words to ensure complete coverage
                - Include extensive analysis and discussion of all relevant aspects
                - Break down complex topics into detailed subsections
                - Provide rich examples and case studies where applicable
                - Explore historical context, current state, and future implications
                - Address multiple perspectives and viewpoints
                - Support all claims with evidence from the sources
                - Use clear topic transitions and maintain logical flow
                - Ensure proper citation of sources throughout
                - Provide tables if relevant.
                
                Start the report immediately without any additional formatting or preamble.
                Format in clean Markdown without code blocks (unless showing code snippets).
                DO NOT include a sources section - it will be added automatically.
                
                Using the following information:
                {json.dumps(report_context, indent=2)}"""
                
                # Save prompt to text file with current date and time
                # Create directory if it doesn't exist
                os.makedirs("prompts", exist_ok=True)
                
                # Create file
                prompt_file = f"prompts/prompt-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                with open(prompt_file, "w") as f:
                    f.write(prompt)
                
                # Generate response
                response = self.report_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 1,
                        'max_output_tokens': 8192,
                    },
                    safety_settings=self.safety_settings
                )
                
                print(response)
                
                # Check for prompt feedback and blocking
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    if any(feedback.block_reason for feedback in response.prompt_feedback):
                        raise ValueError(f"Prompt blocked: {response.prompt_feedback}")
                
                # Verify we have a valid response
                if not response or not response.text:
                    raise ValueError("Invalid response from model")
                
                self.logger.info("Report generation completed")
                return response.text, sources_used
                
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Report generation attempt {attempt + 1} failed: {str(e)}\n"
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All report generation attempts failed: {str(e)}")
                    return None, ""
        
        return None, ""

    async def research(self, query: str) -> str:
        """Main research function that coordinates the entire research process."""
        self.reset_state()
        self.logger.info(f"Starting research: {query}")
        
        research_data = {
            'main_query': query,
            'iterations': [],
            'final_sources': []
        }
        
        while self.research_iterations < self.MAX_ITERATIONS:
            # Combine iteration logs
            self.logger.info(
                f"Iteration {self.research_iterations + 1}: "
                f"Processing {len(self.previous_queries)} queries"
            )
            
            # Get search queries for this iteration
            if self.research_iterations == 0:
                # First iteration: generate initial queries
                search_queries = self.generate_subqueries(query, research_data)
                self.previous_queries.update(search_queries)
            else:
                if not new_queries:
                    self.logger.warning("No additional search queries provided")
                    break
                search_queries = [q for q in new_queries if q not in self.previous_queries]
                if not search_queries:
                    self.logger.warning("No new unique queries to process")
                    break
                self.previous_queries.update(search_queries)
            
            self.logger.info(f"Processing {len(search_queries)} queries for iteration {self.research_iterations + 1}")
            
            # Parallel processing of search and content extraction
            async def process_search_batch():
                # Perform searches in parallel
                search_results = await self.batch_web_search(search_queries)
                if not search_results:
                    return None, []
                
                # Rank results in parallel
                ranked_results = self.rank_new_results(query, search_results)
                
                # Remove duplicates with parallel processing
                seen_urls = set()
                unique_ranked_results = []
                
                async def process_result(result):
                    url = result['url']
                    rewritten_url = self.browser_manager.rewrite_url(url)
                    if rewritten_url not in seen_urls:
                        seen_urls.add(rewritten_url)
                        result['url'] = rewritten_url
                        return result
                    return None
                
                # Process results in parallel
                tasks = [process_result(result) for result in ranked_results]
                processed_results = await asyncio.gather(*tasks)
                unique_ranked_results = [r for r in processed_results if r is not None]
                
                return unique_ranked_results, seen_urls
            
            # Execute search batch processing
            unique_ranked_results, seen_urls = await process_search_batch()
            if not unique_ranked_results:
                self.logger.warning("No valid results found in this iteration")
                break
            
            # Prepare URLs for scraping with parallel processing
            url_to_result = {}
            
            async def process_scraping_candidate(result):
                url = result['url']
                if (url not in self.scraped_urls and 
                    result.get('scrape_decision', {}).get('should_scrape', False)):
                    return url, result
                return None
            
            # Process scraping candidates in parallel
            tasks = [
                process_scraping_candidate(result) 
                for result in list(self.all_results.values()) + unique_ranked_results
            ]
            scraping_candidates = await asyncio.gather(*tasks)
            
            # Fix the dictionary comprehension to properly handle the tuple results
            url_to_result = {}
            for item in scraping_candidates:
                if item:  # If we got a valid result
                    url, result = item  # Unpack the tuple
                    if url not in url_to_result or result.get('relevance_score', 0) > url_to_result[url].get('relevance_score', 0):
                        url_to_result[url] = result
            
            # Convert to list and sort by relevance score
            urls_to_scrape = sorted(
                url_to_result.values(),
                key=lambda x: x.get('relevance_score', 0),
                reverse=True
            )[:15]  # Take top 15
            
            iteration_data = {
                'queries_used': search_queries,
                'findings': []
            }
            
            if urls_to_scrape:
                # Simplify URL scraping logs
                top_url = urls_to_scrape[0].get('url')
                self.logger.info(
                    f"Scraping {len(urls_to_scrape)} URLs "
                    f"(Top URL: {top_url[:60]}{'...' if len(top_url) > 60 else ''})"
                )
                
                # Create a set of unique URLs to scrape
                urls_to_extract = {r['url'] for r in urls_to_scrape}
                
                # Use browser manager for content extraction
                contents = await self.browser_manager.batch_extract_content(list(urls_to_extract))
                
                # Process extracted content in parallel
                async def process_content(result):
                    url = result['url']
                    content = contents.get(url, '')
                    if not content:
                        return None
                        
                    self.scraped_urls.add(url)
                    rewritten_url = self.browser_manager.rewrite_url(url)
                    
                    # Remove individual content storage logs
                    if url == top_url:
                        content_to_store = content[:20000]
                    else:
                        scrape_level = result.get('scrape_decision', {}).get('scrape_level', 'MEDIUM')
                        char_limit = self.get_scrape_limit(scrape_level)
                        content_to_store = content[:char_limit]
                    
                    finding = {
                        'source': rewritten_url,
                        'content': content_to_store,
                        'relevance_score': result.get('relevance_score', 0),
                        'is_top_result': url == top_url,
                        'scrape_level': result.get('scrape_decision', {}).get('scrape_level', 'MEDIUM')
                    }
                    
                    # Update all_results
                    result['url'] = rewritten_url
                    self.all_results[rewritten_url] = result
                    
                    return finding
                
                # Process content in parallel
                content_tasks = [process_content(result) for result in urls_to_scrape]
                findings = await asyncio.gather(*content_tasks)
                
                # Filter out None values and add to iteration data
                valid_findings = [f for f in findings if f is not None]
                iteration_data['findings'].extend(valid_findings)
                research_data['final_sources'].extend(valid_findings)
            
            research_data['iterations'].append(iteration_data)
            
            # Create comprehensive analysis context
            analysis_context = {
                'main_query': query,
                'current_iteration': self.research_iterations,
                'previous_queries': list(self.previous_queries),
                'high_ranking_urls': self.high_ranking_urls,
                'current_findings': iteration_data['findings'],
                'all_iterations': research_data['iterations'],
                'total_sources': len(research_data['final_sources']),
                'unscraped_urls': [
                    {'url': url, 'score': data.get('relevance_score', 0)}
                    for url, data in self.all_results.items()
                    if url not in self.scraped_urls
                ]
            }
            
            # Analyze research state
            need_more_research, explanation, new_queries, _ = self.analyze_research_state(
                query, analysis_context
            )
            
            if need_more_research and self.research_iterations < self.MAX_ITERATIONS - 1:
                self.logger.info(f"Continuing research - Iteration {self.research_iterations + 2}")
                self.research_iterations += 1
                continue
            else:
                self.logger.info(
                    "Research complete: " + 
                    ("sufficient information gathered" if not need_more_research else "maximum iterations reached")
                )
                # Add new log message for report generation decision
                self.logger.info("Moving to final report generation phase...")
                break
        
        if not research_data['final_sources']:
            self.logger.warning("No sources were successfully scraped - continuing with limited information")
            # Add a placeholder finding to ensure research data isn't empty
            research_data['final_sources'].append({
                'source': 'No sources available',
                'content': 'No content could be successfully retrieved.',
                'relevance_score': 0,
                'is_top_result': False,
                'scrape_level': 'LOW'
            })
            research_data['iterations'].append({
                'queries_used': list(self.previous_queries),
                'findings': research_data['final_sources']
            })

        # Generate and save streaming report
        self.logger.info("Generating final report with streaming...")
        report_generator, sources_used = await self.generate_report(
            query, research_data, ""
        )
        
        if report_generator:
            report_file = self.save_report_streaming(
                query, report_generator, sources_used
            )
            if report_file:
                self.logger.info(f"Report has been generated and saved to: {report_file}")
                return f"Report has been generated and saved to: {report_file}"
        
        return "Error: Failed to generate report. Please try again."

def main():
    """Run the research agent with proper async handling."""
    async def run_research():
        async with DeepResearchAgent() as agent:
            query = input("Enter your research query: ")
            result = await agent.research(query)
            print("\nResearch Result:")
            print(result)

    asyncio.run(run_research())

if __name__ == "__main__":
    main() 