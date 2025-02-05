import os
import json
from typing import List, Dict, Optional, Set, Tuple
import google.generativeai as genai
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time
from collections import defaultdict
import logging
import colorlog
from datetime import datetime
import re
from urllib.parse import urlparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from functools import partial
import httplib2

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
        # Initialize different models for different tasks
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Default model for general tasks
        self.ranking_model = genai.GenerativeModel('gemini-1.5-flash')  # Specific model for ranking
        self.analysis_model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')  # Model for analysis
        self.report_model = genai.GenerativeModel('gemini-exp-1206')  # Model for final report generation
        
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
        self.MAX_ITERATIONS = 3
        self.system_context = system_context
        self.total_tokens = 0  # Track total tokens used

    async def __aenter__(self):
        """Async initialization when entering context."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context."""
        await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'context'):
            await self.context.close()
        if hasattr(self, 'browser'):
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()

    async def initialize(self):
        """Initialize async resources."""
        # Initialize playwright with enhanced stealth mode
        self.playwright = await async_playwright().start()
        
        # Enhanced browser launch options
        browser_args = [
            '--disable-blink-features=AutomationControlled',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--disable-gpu',
            '--window-size=1920,1080',
            '--enable-javascript',
            '--accept-cookies'
        ]
        
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=browser_args
        )
        
        # Enhanced context options
        context_options = {
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'viewport': {'width': 1920, 'height': 1080},
            'device_scale_factor': 1,
            'has_touch': False,
            'is_mobile': False,
            'java_script_enabled': True,
            'bypass_csp': True,
            'ignore_https_errors': True,
            'accept_downloads': True,
            'extra_http_headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'DNT': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cookie': 'cookieconsent_status=allow'
            }
        }
        
        self.context = await self.browser.new_context(**context_options)
        
        # Enhanced stealth scripts
        await self.context.add_init_script("""
            // Override common detection methods
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5].map(() => ({length: 0}))});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
            
            // Add Chrome runtime
            window.chrome = {
                runtime: {},
                app: {},
                csi: () => {},
                loadTimes: () => {}
            };
            
            // Override permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({state: Notification.permission}) :
                originalQuery(parameters)
            );
            
            // Add missing browser features
            window.Notification = window.Notification || {};
            window.Notification.permission = 'denied';
            
            // Prevent iframe detection
            Object.defineProperty(window, 'parent', {get: () => window});
            Object.defineProperty(window, 'top', {get: () => window});
        """)

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

    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is from YouTube."""
        parsed = urlparse(url)
        return any(yt in parsed.netloc for yt in ['youtube.com', 'youtu.be'])

    def rewrite_url(self, url: str) -> str:
        """Rewrite URLs based on defined rules."""
        parsed = urlparse(url)
        
        # Reddit URL rewrite
        if 'reddit.com' in parsed.netloc:
            # Replace reddit.com with rl.bloat.cat while keeping the rest of the URL structure
            return url.replace('reddit.com', 'redlib.kylrth.com/')
            
        return url

    def should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped."""
        return (
            url in self.blacklisted_urls or
            self.is_youtube_url(url) or
            any(ext in url for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx'])
        )

    def generate_subqueries(self, main_query: str, research_state: Optional[Dict] = None) -> List[str]:
        """Generate sub-queries using AI to explore different aspects of the main query."""
        context = ""
        if research_state and self.previous_queries:
            context = f"""
            Previously used queries: {json.dumps(list(self.previous_queries), indent=2)}
            Current research state: {json.dumps(research_state, indent=2)}
            Based on the above context and gaps in current research, """
        
        prompt = f"""{self.system_context}
        {context}Given the main query: '{main_query}', generate 5-10 short, focused search queries 
        that would be effective for Google search.
        
        Guidelines:
        - Keep queries generic for current/latest things
        - Keep each query under 5-6 words
        - Use exact phrases in quotes when needed
        
        Format: One query per line, starting with a number."""
        
        self.logger.info("Generating search queries...")
        
        try:
            response = self.model.generate_content(prompt)
            self.logger.debug(f"AI Response:\n{response.text}")
            
            subqueries = [q.strip().split('.', 1)[1].strip() 
                         for q in response.text.split('\n') 
                         if q.strip() and any(c.isdigit() for c in q)]
            
            self.logger.info(f"Generated {len(subqueries)} queries:")
            for q in subqueries:
                self.logger.info(f"Query: {q}")
            
            # Don't add to previous_queries here, let the research method handle it
            return subqueries[:10]
        except Exception as e:
            self.logger.error(f"Error generating queries: {e}")
            return [main_query]

    async def batch_web_search(self, queries: List[str], num_results: int = 8) -> List[Dict]:
        """Perform multiple web searches in parallel."""
        self.logger.info(f"Batch searching {len(queries)} queries...")
        
        def search_single(query: str) -> List[Dict]:
            """Synchronous search function to run in thread pool."""
            try:
                # Add retry logic and better error handling
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Add SSL verification settings
                        http = httplib2.Http(timeout=30)
                        http.disable_ssl_certificate_validation = True
                        
                        results = self.search_engine.list(
                            q=query,
                            cx=self.search_engine_id,
                            num=num_results
                        ).execute(http=http)
                        
                        if not results or 'items' not in results:
                            self.logger.warning(f"No results found for query: {query}")
                            return []
                        
                        search_results = []
                        for item in results.get('items', []):
                            try:
                                url = item.get('link', '')
                                if not url or self.should_skip_url(url):
                                    continue
                                    
                                result = {
                                    'title': item.get('title', ''),
                                    'url': url,  # Remove rewrite, use original URL
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
                        self.logger.warning(f"Search attempt {attempt + 1} failed: {retry_error}")
                        time.sleep(2)  # Wait before retry
                        
            except Exception as e:
                self.logger.error(f"Search error for query '{query}': {str(e)}")
                return []
        
        # Run searches in thread pool with smaller batch size
        loop = asyncio.get_running_loop()
        batch_size = 3  # Process queries in smaller batches
        all_results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                search_futures = [
                    loop.run_in_executor(executor, search_single, query)
                    for query in batch_queries
                ]
                batch_results = await asyncio.gather(*search_futures)
                
                # Process batch results
                for query, query_results in zip(batch_queries, batch_results):
                    for result in query_results:
                        url = result['url']
                        if url in self.all_results:
                            self.all_results[url]['source_queries'].append(query)
                        else:
                            self.all_results[url] = result
                            all_results.append(result)
                
                # Add small delay between batches
                if i + batch_size < len(queries):
                    await asyncio.sleep(1)
        
        self.logger.info(f"Found {len(all_results)} unique results across all queries")
        return all_results

    async def extract_content_with_retry(self, url: str, max_retries: int = 2) -> str:
        """Extract content from a webpage with simplified retry logic."""
        self.logger.info(f"Extracting content: {url}")
        
        for attempt in range(max_retries):
            page = None
            try:
                page = await self.context.new_page()
                
                # Try to load the page
                try:
                    response = await page.goto(
                        url, 
                        wait_until='domcontentloaded',
                        timeout=10000
                    )
                    
                    if not response.ok:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        return ""
                        
                except PlaywrightTimeout:
                    self.logger.warning(f"Timeout loading {url}")
                    return ""
                
                # Simple content extraction
                content = await page.evaluate("""
                    () => {
                        // Try to get main content first
                        const article = document.querySelector('article, [role="main"], main, .content, .post');
                        if (article) {
                            const text = article.innerText;
                            if (text.length >= 100) return text;
                        }
                        
                        // If no main content, get all meaningful text
                        const textContent = [];
                        const elements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, td, th, div > text');
                        
                        elements.forEach(elem => {
                            // Skip if element is hidden or part of navigation/footer
                            const style = window.getComputedStyle(elem);
                            if (style.display === 'none' || style.visibility === 'hidden') return;
                            if (elem.closest('nav, footer, header, aside')) return;
                            
                            const text = elem.innerText.trim();
                            // Only add if text is meaningful (not just a number or single word)
                            if (text.length >= 20 || (text.includes(' ') && text.length >= 10)) {
                                textContent.push(text);
                            }
                        });
                        
                        return textContent.join('\\n\\n');
                    }
                """)
                
                if content and len(content) >= 100:
                    return content.strip()
                
                return ""
                
            except Exception as e:
                self.logger.warning(f"Extraction error for {url}: {str(e)}")
                return ""
            finally:
                if page:
                    try:
                        await page.close()
                    except:
                        pass
        
        return ""

    async def batch_extract_content(self, urls: List[str], max_concurrent: int = 3) -> Dict[str, str]:
        """Extract content from multiple URLs in parallel with concurrency limit."""
        self.logger.info(f"Batch extracting content from {len(urls)} URLs...")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_semaphore(url: str) -> Tuple[str, str]:
            async with semaphore:
                content = await self.extract_content_with_retry(url)
                return url, content
        
        tasks = [extract_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def rank_new_results(self, main_query: str, new_results: List[Dict]) -> List[Dict]:
        """Rank only new search results based on relevance using AI."""
        if not new_results:
            return []

        self.logger.info(f"Ranking {len(new_results)} new URLs")

        prompt = f"""{self.system_context}
        For the query '{main_query}', rank these new search results by relevance.
        Respond with ONLY scores (0-0.99) for each URL, one per line.
        IMPORTANT RULES:
        - Only give a score of 1.0 if the content PERFECTLY matches the query intent
        - All other scores MUST be unique (no ties) and between 0 and 0.99
        - Use enough decimal places to ensure uniqueness
        - Higher scores for:
          * Official sources
          * Recent content (based on current date: {self.current_date})
          * Comprehensive information
          * Reliable domains
        
        Format:
        [url] [score]
        
        URLs to rank:
        """ + "\n".join([
            f"{data['url']}\nTitle: {data['title']}\nSnippet: {data['snippet']}"
            for data in new_results
        ])

        try:
            response = self.ranking_model.generate_content(prompt)
            self.logger.debug(f"Ranking response:\n{response.text}")
            
            # Parse rankings and verify uniqueness
            rankings = {}
            used_scores = set()
            
            for line in response.text.strip().split('\n'):
                try:
                    url, score = line.rsplit(' ', 1)
                    url = url.strip()
                    score = float(score)
                    
                    # Ensure score is valid
                    if score != 1.0 and score in used_scores and score > 0.99:
                        self.logger.warning(f"Duplicate score {score} found, adjusting slightly")
                        while score in used_scores:
                            score = max(0, min(0.99, score - 0.001))
                    
                    used_scores.add(score)
                    rankings[url] = score

                    # Track high-ranking URLs (score > 0.6)
                    if score > 0.6:
                        result = next((r for r in new_results if r['url'] == url), None)
                        if result:
                            self.high_ranking_urls[url] = {
                                'score': score,
                                'title': result['title'],
                                'snippet': result['snippet'],
                                'domain': result['domain'],
                                'source_queries': result['source_queries']
                            }
                except (ValueError, IndexError):
                    continue
            
            # Update scores and sort results
            ranked_results = []
            for result in new_results:
                if result['url'] in rankings:
                    result['relevance_score'] = rankings[result['url']]
                    ranked_results.append(result)
            
            ranked_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            self.logger.info("Top 5 ranked new URLs:")
            for r in ranked_results[:5]:
                self.logger.info(f"{r['url']} (Score: {r.get('relevance_score', 0):.3f})")
            
            return ranked_results
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            return new_results

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

    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage using Playwright."""
        self.logger.info(f"Extracting content: {url}")
        try:
            page = self.context.new_page()
            page.goto(url, wait_until='networkidle', timeout=20000)
            
            # Wait for content to load
            page.wait_for_selector('body', timeout=5000)
            
            # Remove unwanted elements
            page.evaluate("""() => {
                const selectors = ['script', 'style', 'nav', 'header', 'footer', 'iframe', 'aside', 'form'];
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => el.remove());
                });
            }""")
            
            # Extract text from main content elements
            content = page.evaluate("""() => {
                const contentElements = document.querySelectorAll('article, main, .content, .post, p, h1, h2, h3, h4, h5, h6');
                const textContent = [];
                
                contentElements.forEach(elem => {
                    const text = elem.textContent.trim();
                    if (text.length >= 50 && elem.querySelectorAll('a').length < 5) {
                        textContent.push(text);
                    }
                });
                
                return textContent.join(' ');
            }""")
            
            page.close()
            
            self.logger.info(f"Extracted {len(content)} characters")
            return content
        except Exception as e:
            self.logger.error(f"Extraction error: {e}")
            return ""

    def analyze_research_state(self, main_query: str, current_data: Dict) -> Tuple[bool, str, List[str]]:
        """Analyze current research state and decide if more research is needed."""
        self.logger.info("Analyzing research state...")
        
        prompt = f"""{self.system_context}
        Analyze the research on '{main_query}'.
        Current data: {json.dumps(current_data, indent=2)}
        
        Determine if more research is needed and what specific information is missing.
        Also identify any URLs that should be blacklisted (irrelevant, outdated, etc.).
        
        If more research is needed, provide up to 10 COMPLETE search queries that would help fill the gaps.
        These should be actual search queries ready to be used. Keep them simple.
        Make sure to relate the queries to the main query to find extra information about the topic and the results you have already found.
        
        Say NO at any time to end the research and produce a final report. You should be saying NO when the original query is answered, or when you have gathered enough information.
        
        Format response EXACTLY as follows:
        DECISION: [YES/NO]
        BLACKLIST: [List URLs to blacklist, one per line]
        MISSING: [List missing information aspects, one per line]
        SEARCH_QUERIES: [List complete search queries, one per line, max 10]"""
        
        try:
            response = self.analysis_model.generate_content(prompt)  # Use analysis specific model
            self.logger.debug(f"Analysis response:\n{response.text}")
            
            # Parse sections
            decision = False
            blacklist = []
            missing = []
            search_queries = []
            current_section = None
            
            for line in response.text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Handle section headers
                if line.startswith("DECISION:"):
                    decision = "YES" in line.upper()
                    current_section = None
                elif line.startswith("BLACKLIST:"):
                    current_section = "BLACKLIST"
                elif line.startswith("MISSING:"):
                    current_section = "MISSING"
                elif line.startswith("SEARCH_QUERIES:"):
                    current_section = "SEARCH_QUERIES"
                else:
                    # Handle section content
                    if current_section == "BLACKLIST" and line.startswith('http'):
                        blacklist.append(line.strip())
                    elif current_section == "MISSING" and line.startswith('-'):
                        missing.append(line[1:].strip())
                    elif current_section == "SEARCH_QUERIES":
                        # Handle multiple query formats
                        if line.startswith('- '):
                            search_queries.append(line[2:].strip())
                        elif line.strip() and not line.startswith(('DECISION:', 'BLACKLIST:', 'MISSING:', 'SEARCH_QUERIES:')):
                            # Handle numbered or plain queries
                            clean_query = line.split('. ', 1)[-1] if '. ' in line else line
                            search_queries.append(clean_query.strip())
            
            # Update blacklist
            self.blacklisted_urls.update(blacklist)
            
            # Prepare explanation
            explanation = "Missing Information:\n" + "\n".join(f"- {m}" for m in missing)
            explanation += "\n\nSearch Queries to Try:\n" + "\n".join(f"- {q}" for q in search_queries)
            
            return decision, explanation, search_queries
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return False, str(e), []

    def save_report(self, query: str, report: str):
        """Save the report to a markdown file."""
        try:
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Clean query for filename
            clean_query = re.sub(r'[^\w\s-]', '', query).strip().lower()
            clean_query = re.sub(r'[-\s]+', '-', clean_query)
            
            # Create filename with date
            filename = f"reports/{clean_query}-{self.current_date}.md"
            
            # Add metadata to report
            full_report = f"""# Research Report: {query}
Date: {self.current_date}
Location: {self.approximate_location}

{report}"""
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_report)
            
            self.logger.info(f"Report saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")

    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        # Rough estimation: 4 chars per token on average
        return len(text) // 4

    def log_token_usage(self, text: str, operation: str):
        """Log token usage for an operation."""
        tokens = self.count_tokens(text)
        self.total_tokens += tokens
        self.logger.info(f"Token usage for {operation}: {tokens} (Total: {self.total_tokens})")

    def reset_state(self):
        """Reset all state tracking for a new query."""
        self.previous_queries = set()
        self.all_results = {}
        self.high_ranking_urls = {}
        self.blacklisted_urls = set()
        self.scraped_urls = set()
        self.research_iterations = 0
        self.total_tokens = 0
        self.logger.info("Reset research state")

    async def research(self, query: str) -> str:
        """Main research function that coordinates the entire research process."""
        # Reset state for new query
        self.reset_state()
        
        self.logger.info(f"Starting research on: {query}")
        
        research_data = {
            'main_query': query,
            'iterations': [],
            'final_sources': []
        }
        
        while self.research_iterations < self.MAX_ITERATIONS:
            self.logger.info(f"Starting research iteration {self.research_iterations + 1}")
            
            # Get search queries for this iteration
            if self.research_iterations == 0:
                # First iteration: generate initial queries
                search_queries = self.generate_subqueries(query, research_data)
                self.previous_queries.update(search_queries)  # Add to previous queries after generation
            else:
                # Use the search queries from the previous analysis
                if not new_queries:  # new_queries comes from previous iteration's analysis
                    self.logger.warning("No additional search queries provided")
                    break
                # Filter out previously used queries
                search_queries = [q for q in new_queries if q not in self.previous_queries]
                if not search_queries:
                    self.logger.warning("No new unique queries to process")
                    break
                # Update previous queries with the new ones
                self.previous_queries.update(search_queries)
            
            self.logger.info(f"Processing {len(search_queries)} queries for iteration {self.research_iterations + 1}")
            
            # Batch process all searches
            search_results = await self.batch_web_search(search_queries)
            if not search_results:
                self.logger.warning("No search results found")
                break
            
            # Rank new results before adding to all_results
            ranked_new_results = self.rank_new_results(query, search_results)
            
            # Filter out already scraped URLs and get top 8
            new_urls = [r for r in ranked_new_results if r['url'] not in self.scraped_urls][:8]
            if not new_urls:
                self.logger.warning("No new URLs to scrape")
                break
                
            iteration_data = {
                'queries_used': search_queries,
                'findings': []
            }
            
            # Batch process content extraction for new URLs only
            contents = await self.batch_extract_content([r['url'] for r in new_urls])
            
            # Update scraped URLs and process content
            for result in new_urls:
                url = result['url']
                content = contents.get(url, '')
                if content:
                    self.scraped_urls.add(url)  # Mark as scraped
                    
                    # Apply URL rewrite after content extraction
                    rewritten_url = self.rewrite_url(url)
                    
                    finding = {
                        'source': rewritten_url,
                        'content': content[:6000],
                        'relevance_score': result.get('relevance_score', 0)
                    }
                    iteration_data['findings'].append(finding)
                    research_data['final_sources'].append(finding)
                    self.log_token_usage(content[:6000], f"Content from {url}")
                    
                    # Add to all_results after successful scraping
                    result['url'] = rewritten_url  # Update URL to rewritten version
                    self.all_results[rewritten_url] = result
            
            research_data['iterations'].append(iteration_data)
            
            # Create comprehensive analysis context
            analysis_context = {
                'main_query': query,
                'current_iteration': self.research_iterations,
                'previous_queries': list(self.previous_queries),
                'high_ranking_urls': self.high_ranking_urls,
                'current_findings': iteration_data['findings'],
                'all_iterations': research_data['iterations'],
                'total_sources': len(research_data['final_sources'])
            }
            
            # Analyze if more research is needed - store results for next iteration
            need_more_research, explanation, new_queries = self.analyze_research_state(query, analysis_context)
            self.log_token_usage(explanation, "Research state analysis")
            
            if need_more_research and self.research_iterations < self.MAX_ITERATIONS - 1:
                self.logger.info(f"More research needed:\n{explanation}")
                self.research_iterations += 1
                continue
            else:
                if not need_more_research:
                    self.logger.info("Research complete - sufficient information gathered")
                else:
                    self.logger.warning("Maximum iterations reached")
                break
        
        if not research_data['final_sources']:
            return "Error: No valid information could be gathered. Please try again or modify the query."
        
        # Generate and save report
        self.logger.info("Generating final report...")
        report = self.generate_report(query, research_data)
        self.log_token_usage(report, "Final report generation")
        self.save_report(query, report)
        
        self.logger.info(f"Total tokens used in research: {self.total_tokens}")
        return report

    def generate_report(self, main_query: str, research_data: Dict) -> str:
        """Generate a comprehensive report from the gathered research data."""
        # Prepare enhanced context with high-ranking URLs
        report_context = {
            'main_query': main_query,
            'research_data': research_data,
            'high_ranking_sources': self.high_ranking_urls,
            'total_sources_analyzed': len(self.all_results),
            'total_high_ranking_sources': len(self.high_ranking_urls),
            'research_iterations': self.research_iterations,
            'total_queries_used': len(self.previous_queries)
        }
        
        prompt = f"""Generate a comprehensive research report on: '{main_query}'
        Using the following information:
        {json.dumps(report_context, indent=2)}
        
        Important Guidelines:
        - Focus on information from high-ranking sources (score > 0.6)
        - Cross-reference information across multiple sources
        - Highlight any conflicting information found
        - Include citation links using the source URLs
        - If you have a LOT of information, this report should be long, detailed and comprehensive.
        
        Organize the report with clear sections, including:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Sources and Citations (prioritize high-ranking sources)
        5. Research Methodology (including iterations and refinements)
        
        Make it detailed yet easy to understand.
        Format the report in Markdown."""
        
        try:
            response = self.report_model.generate_content(prompt)  # Use report specific model
            self.logger.info("Report generated successfully")
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Error generating report. Please try again.\nError: {str(e)}"

def main():
    """Run the research agent with proper async handling."""
    async def run_research():
        async with DeepResearchAgent() as agent:
            query = input("Enter your research query: ")
            report = await agent.research(query)
            print("\nResearch Report:")
            print(report)

    asyncio.run(run_research())

if __name__ == "__main__":
    main() 