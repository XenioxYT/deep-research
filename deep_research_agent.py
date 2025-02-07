import os
import json
from typing import List, Dict, Optional, Set, Tuple
import google.generativeai as genai
from test_search import build
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
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from vertexai.preview import tokenization

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
        self.MAX_ITERATIONS = 9
        self.system_context = system_context
        self.total_tokens = 0  # Track total tokens used
        
        # Initialize tokenizer for Gemini model
        self.model_name = "gemini-1.5-flash-002"
        self.tokenizer = tokenization.get_tokenizer_for_model(self.model_name)
        self.token_usage_by_operation = defaultdict(int)
        self.content_tokens = 0  # Track tokens from stored content separately

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
        self.logger.info("Analyzing query and generating search queries...")
        
        context = ""
        if research_state and self.previous_queries:
            context = f"""
            Previously used queries: {json.dumps(list(self.previous_queries), indent=2)}
            Current research state: {json.dumps(research_state, indent=2)}
            Based on the above context and gaps in current research, """
        
        prompt = f"""{self.system_context}
        {context}Analyze this query and generate appropriate search queries.

        Query: '{main_query}'

        First, determine if this is a SIMPLE query that needs no additional research:
        - Basic arithmetic (e.g., "what is 2+2")
        - Single fact lookups (e.g., "capital of France")
        - Simple definitions (e.g., "what is a tree")
        - Basic conversions (e.g., "10 km to miles")
        - Yes/no questions with obvious answers

        Format your response EXACTLY as follows:
        TYPE: [SIMPLE/COMPLEX]
        REASON: [One sentence explanation]
        QUERIES:
        [If SIMPLE: Only list the original query
        If COMPLEX: Generate 5-10 focused search queries, one per line, starting with numbers
        - Keep queries generic for current/latest things
        - Keep each query under 5-6 words
        - Keep queries open ended, not too specific]"""

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
                            
                            # Validate query
                            if query and len(query.split()) <= 6 and len(query) >= 3:
                                subqueries.append(query)
                        except Exception as e:
                            self.logger.warning(f"Error processing query line '{line}': {e}")
                            continue
            
            # Always include the main query for complex queries
            if query_type == "COMPLEX" and main_query not in subqueries:
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

    async def batch_web_search(self, queries: List[str], num_results: int = 8) -> List[Dict]:
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
                        timeout=15000  # Increased timeout
                    )
                    
                    if not response.ok:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        return ""
                        
                except PlaywrightTimeout:
                    self.logger.warning(f"Timeout loading {url}")
                    return ""
                
                # Enhanced content extraction with better text handling
                content = await page.evaluate("""
                    () => {
                        // Helper function to clean text
                        const cleanText = (text) => {
                            return text.replace(/\\s+/g, ' ').trim();
                        };
                        
                        // Helper to check if element is visible
                        const isVisible = (elem) => {
                            const style = window.getComputedStyle(elem);
                            return style.display !== 'none' && 
                                   style.visibility !== 'hidden' && 
                                   style.opacity !== '0' &&
                                   elem.offsetWidth > 0 &&
                                   elem.offsetHeight > 0;
                        };
                        
                        // Get all meaningful content
                        const contentSections = [];
                        
                        // Try to get main content first
                        const mainContent = document.querySelector('article, [role="main"], main, .content, .post');
                        if (mainContent && isVisible(mainContent)) {
                            const text = cleanText(mainContent.innerText);
                            if (text.length >= 100) {
                                contentSections.push(text);
                            }
                        }
                        
                        // Get content from all meaningful elements
                        const elements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, td, th, div:not(:empty)');
                        
                        elements.forEach(elem => {
                            // Skip if element is hidden or part of navigation/footer/etc
                            if (!isVisible(elem) || elem.closest('nav, footer, header, aside, .menu, .navigation')) {
                                return;
                            }
                            
                            // Skip elements with too many links (likely navigation)
                            if (elem.querySelectorAll('a').length > elem.textContent.split(' ').length / 3) {
                                return;
                            }
                            
                            const text = cleanText(elem.innerText);
                            // Only add if text is meaningful
                            if (text.length >= 20 || (text.includes(' ') && text.length >= 10)) {
                                contentSections.push(text);
                            }
                        });
                        
                        // Join all content with proper spacing
                        return contentSections.join('\\n\\n');
                    }
                """)
                
                if content:
                    # Clean content server-side as well
                    content = re.sub(r'\s+', ' ', content).strip()
                    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
                    
                    if len(content) >= 100:  # Only return if we got meaningful content
                        return content
                
                return ""
                
            except Exception as e:
                self.logger.warning(f"Extraction error for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Wait before retry
                return ""
            finally:
                if page:
                    try:
                        await page.close()
                    except:
                        pass
        
        return ""

    async def batch_extract_content(self, urls: List[str], max_concurrent: int = 8) -> Dict[str, str]:
        """Extract content from multiple URLs in parallel with enhanced concurrency."""
        self.logger.info(f"Extracting content from {len(urls)} URLs")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create a pool of browser pages
        page_pool = []
        for _ in range(max_concurrent):
            page = await self.context.new_page()
            page_pool.append(page)
        
        async def extract_with_page_pool(url: str) -> Tuple[str, str]:
            """Extract content using a page from the pool."""
            async with semaphore:
                page = None
                try:
                    # Get a page from the pool
                    page = page_pool.pop()
                    
                    # Enhanced stealth settings per page
                    await page.add_init_script("""
                        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                    """)
                    
                    # Configure page timeouts
                    page.set_default_navigation_timeout(20000)
                    page.set_default_timeout(5000)
                    
                    # Try to load the page with retry logic
                    content = ""
                    max_retries = 1
                    base_delay = 1
                    
                    for attempt in range(max_retries):
                        try:
                            # Navigate with enhanced options
                            response = await page.goto(
                                url,
                                wait_until='domcontentloaded',
                                timeout=5000
                            )
                            
                            if not response or not response.ok:
                                if attempt < max_retries - 1:
                                    delay = base_delay * (2 ** attempt)
                                    await asyncio.sleep(delay)
                                    continue
                                return url, ""
                            
                            # Wait for content to be available
                            await page.wait_for_selector('body', timeout=5000)
                            
                            # Enhanced content extraction with better text handling
                            content = await page.evaluate("""
                                () => {
                                    // Helper to clean text
                                    const cleanText = (text) => {
                                        return text.replace(/\\s+/g, ' ').trim();
                                    };
                                    
                                    // Helper to check visibility
                                    const isVisible = (elem) => {
                                        const style = window.getComputedStyle(elem);
                                        return style.display !== 'none' && 
                                               style.visibility !== 'hidden' && 
                                               style.opacity !== '0' &&
                                               elem.offsetWidth > 0 &&
                                               elem.offsetHeight > 0;
                                    };
                                    
                                    // Get main content first
                                    const contentSections = new Set();
                                    const mainContent = document.querySelector('article, [role="main"], main, .content, .post');
                                    if (mainContent && isVisible(mainContent)) {
                                        contentSections.add(cleanText(mainContent.innerText));
                                    }
                                    
                                    // Get content from meaningful elements
                                    const elements = document.querySelectorAll(
                                        'p, h1, h2, h3, h4, h5, h6, li, td, th, div:not(:empty)'
                                    );
                                    
                                    elements.forEach(elem => {
                                        if (!isVisible(elem) || 
                                            elem.closest('nav, footer, header, aside, .menu, .navigation')) {
                                            return;
                                        }
                                        
                                        // Skip elements with too many links
                                        if (elem.querySelectorAll('a').length > 
                                            elem.textContent.split(' ').length / 3) {
                                            return;
                                        }
                                        
                                        const text = cleanText(elem.innerText);
                                        if (text.length >= 20 || (text.includes(' ') && text.length >= 10)) {
                                            contentSections.add(text);
                                        }
                                    });
                                    
                                    return Array.from(contentSections).join('\\n\\n');
                                }
                            """)
                            
                            if content:
                                # Clean content server-side
                                content = re.sub(r'\s+', ' ', content).strip()
                                content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
                                break
                                
                        except PlaywrightTimeout:
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                await asyncio.sleep(delay)
                                continue
                            return url, ""
                            
                        except Exception as e:
                            self.logger.warning(f"Extraction error for {url} (attempt {attempt + 1}): {str(e)}")
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                await asyncio.sleep(delay)
                                continue
                            return url, ""
                            
                finally:
                    if page:
                        # Return page to pool
                        page_pool.append(page)
                
                return url, content
        
        try:
            # Process URLs in parallel
            tasks = [extract_with_page_pool(url) for url in urls]
            results = await asyncio.gather(*tasks)
            
            # Clean up page pool
            for page in page_pool:
                await page.close()
            
            return dict(results)
            
        except Exception as e:
            self.logger.error(f"Batch extraction error: {e}")
            # Clean up page pool on error
            for page in page_pool:
                try:
                    await page.close()
                except:
                    pass
            return {}

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
                    should_scrape = scrape_decision.upper() == 'YES' and scrape_count < 10
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
        2. For queries that can be fully answered with current findings, say NO
        3. For queries needing more depth/verification:
           - On iteration 0-2: Say YES if information is missing
           - On iteration 3: Only say YES if crucial information is missing
           - On iteration 4+: Strongly lean towards NO unless absolutely critical information is missing
        4. Consider the query ANSWERED when you have:
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
        SEARCH_QUERIES: [List complete search queries, one per line, max 10. Search formatting and quotes are allowed. These queries should be specific to the information you are looking for.]
        SCRAPE_NEXT: [List URLs to scrape in next iteration, one per line, in format: URL | LOW/MEDIUM/HIGH]
        REPORT_STRUCTURE: [A complete, customized report structure and guidelines based on the query type and findings. This should include:
        1. Required sections and their order
        2. What to include in each section
        3. Specific formatting guidelines
        4. Try to include a table or information where relevant
        5. Any special considerations for this topic
        The structure should be tailored to the specific query type (e.g., product analysis, historical research, current events, etc.)]"""
        
        try:
            response = self.analysis_model.generate_content(prompt)
            
            # Parse sections
            decision = False
            query_type = "COMPLEX"  # Default to complex
            blacklist = []
            missing = []
            search_queries = []
            report_structure = ""
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
            
            # For simple queries, ensure we have a report structure
            if query_type == "SIMPLE" and not report_structure:
                report_structure = """
                # Answer to: {query}
                
                ## Direct Answer
                Provide the simple, factual answer to the query.
                
                ## Brief Explanation (if needed)
                Optional brief explanation of the answer, if relevant.
                
                ## Additional Context (if applicable)
                Any relevant context or related information, if appropriate.
                """.strip()
            
            return decision, explanation, search_queries, report_structure
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
                
                # Enhanced report context with version tracking
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

                # Add version-specific context
                if 'initial_report' in research_data:
                    report_context.update({
                        'is_revision': True,
                        'initial_report': research_data['initial_report'],
                        'review_feedback': research_data.get('review_feedback', {}),
                        'previous_versions': research_data.get('report_versions', [])
                    })
                
                prompt = f"""Generate a comprehensive research report on: '{main_query}'
                Using the following information:
                {json.dumps(report_context, indent=2)}
                
                Follow this custom report structure and guidelines:
                {report_structure}

                {'This is a REVISED report. You must:' if 'is_revision' in report_context else ''}
                {'1. Address the review feedback and missing aspects identified' if 'is_revision' in report_context else ''}
                {'2. Incorporate new findings while maintaining relevant information from the initial report' if 'is_revision' in report_context else ''}
                {'3. Highlight significant changes or additions from the initial version' if 'is_revision' in report_context else ''}
                {'4. Provide a synthesis of all findings' if 'is_revision' in report_context else ''}
                
                Include an "Opinion" section at the end of the report. This should focus on your views on the topic and the sources used to form those views.
                The report should be long and detailed, and should include all the information from the sources used.
                Contradictions in sources should be noted and explained, and the report should provide a conclusion that takes into account the contradictions.
                
                Markdown Formatting Guidelines:
                - Use section headers with #, ##, and ###
                - Use **bold** for emphasis on key points
                - Format numbers naturally with proper thousands separators
                - DO NOT place references in the form [1, 2, 3, 4, 5, 6, 9]. Always do [1][2][3] etc.
                - You can use your own knowledge to add additional information to the report, however you must say when you have done so and mention that you might hallucinate. Be sure to mention when and where you have used your own knowledge!
                - You can use LATEX formatting. ALWAYS wrap mathematical equations in $$[latex]$$
                
                Start the report immediately after this prompt without any additional formatting or preamble.
                Format in clean Markdown without code blocks (unless for code snippets).
                
                DO NOT include a sources section - it will be added automatically."""
                
                # Generate response
                response = self.report_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'max_output_tokens': 8192,
                    },
                    safety_settings=self.safety_settings
                )
                
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

    async def review_report(self, query: str, report_text: str, research_data: Dict) -> Tuple[bool, str, List[str], List[str]]:
        """
        Review the generated report for completeness and quality.
        Returns: (is_approved, review_notes, missing_aspects, additional_queries)
        """
        # Use a separate model for review to get a fresh perspective
        review_model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            safety_settings=self.safety_settings
        )
        
        # Enhanced review prompt with more specific criteria
        prompt = f"""Review this research report on: '{query}'
        
        Report Text:
        {report_text}
        
        Research Context:
        - Total sources analyzed: {len(self.all_results)}
        - High-ranking sources used: {len(self.high_ranking_urls)}
        - Research iterations: {self.research_iterations}
        - Previous queries used: {list(self.previous_queries)}
        
        Analyze the report based on these specific criteria:

        1. Completeness (30 points):
        - Are all aspects of the query addressed?
        - Is there sufficient background information?
        - Are all relevant perspectives covered?

        2. Depth (25 points):
        - Is the analysis thorough and detailed?
        - Are complex concepts well explained?
        - Is there adequate supporting evidence?

        3. Evidence Quality (20 points):
        - Are claims supported by credible sources?
        - Is source information recent and relevant?
        - Are contradictions or conflicts addressed?

        4. Structure and Clarity (15 points):
        - Is the report well-organized?
        - Is the writing clear and professional?
        - Are sections properly connected?

        5. Technical Accuracy (10 points):
        - Are facts and figures accurate?
        - Are technical terms used correctly?
        - Are calculations or data interpretations sound?

        Format your response EXACTLY as:
        APPROVED: [YES/NO]
        SCORE: [0-100]
        REVIEW:
        [Detailed review notes with specific examples]
        
        SCORES_BREAKDOWN:
        Completeness: [X/30]
        Depth: [X/25]
        Evidence: [X/20]
        Structure: [X/15]
        Accuracy: [X/10]
        
        MISSING_ASPECTS:
        [List specific missing information or aspects that need more research, one per line]
        
        SEARCH_QUERIES:
        [If not approved, list of 15 specific search queries to find the missing information]
        
        IMPROVEMENT_SUGGESTIONS:
        [List specific suggestions for improving the report]
        """
        
        try:
            response = review_model.generate_content(prompt)
            
            # Parse review results with enhanced error handling
            approved = False
            review_notes = ""
            missing_aspects = []
            search_queries = []
            current_section = None
            improvement_suggestions = []
            
            if not response or not response.text:
                raise ValueError("Empty response from review model")
            
            for line in response.text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    if line.startswith('APPROVED:'):
                        approved = 'YES' in line.upper()
                    elif line.startswith('SCORE:'):
                        score = int(line.split(':')[1].strip().split('/')[0])
                        # Auto-approve if score is very high
                        if score >= 90:
                            approved = True
                    elif line.startswith('MISSING_ASPECTS:'):
                        current_section = 'missing'
                    elif line.startswith('SEARCH_QUERIES:'):
                        current_section = 'queries'
                    elif line.startswith('REVIEW:'):
                        current_section = 'review'
                    elif line.startswith('IMPROVEMENT_SUGGESTIONS:'):
                        current_section = 'improvements'
                    elif line and not line.startswith(('APPROVED:', 'SCORE:', 'SCORES_BREAKDOWN:', 'MISSING_ASPECTS:', 'SEARCH_QUERIES:', 'REVIEW:', 'IMPROVEMENT_SUGGESTIONS:')):
                        if current_section == 'missing':
                            if line.strip('- '):  # Only add non-empty lines
                                missing_aspects.append(line.strip('- '))
                        elif current_section == 'queries':
                            if line.strip('- '):  # Only add non-empty lines
                                search_queries.append(line.strip('- '))
                        elif current_section == 'review':
                            review_notes += line + '\n'
                        elif current_section == 'improvements':
                            if line.strip('- '):
                                improvement_suggestions.append(line.strip('- '))
                
                except Exception as parse_error:
                    self.logger.warning(f"Error parsing review line '{line}': {parse_error}")
                    continue
            
            # Add improvement suggestions to review notes
            if improvement_suggestions:
                review_notes += "\n\nImprovement Suggestions:\n"
                review_notes += "\n".join(f"- {suggestion}" for suggestion in improvement_suggestions)
            
            # Ensure we have some output
            if not review_notes.strip():
                review_notes = "Review completed but no specific notes were generated."
            
            # Log review results
            self.logger.info(f"Report review completed - Approved: {approved}")
            if not approved:
                self.logger.info(f"Missing aspects: {len(missing_aspects)}")
                self.logger.info(f"Additional queries: {len(search_queries)}")
            
            return approved, review_notes.strip(), missing_aspects, search_queries
            
        except Exception as e:
            self.logger.error(f"Report review error: {e}")
            # Return conservative default - not approved with error note
            return False, f"Review failed due to error: {str(e)}", [], []

    async def perform_research_cycle(self, search_queries: List[str], research_data: Dict) -> Dict:
        """
        Perform a complete research cycle: search, rank, analyze, and extract content.
        Returns the iteration data with findings.
        """
        self.logger.info(f"Starting research cycle with {len(search_queries)} queries")
        
        # Perform batch web search
        search_results = await self.batch_web_search(search_queries)
        if not search_results:
            self.logger.warning("No search results found")
            return None
            
        # Rank the results
        ranked_results = self.rank_new_results(research_data['main_query'], search_results)
        if not ranked_results:
            self.logger.warning("No valid ranked results")
            return None
            
        # Process results and update research data
        iteration_data = {
            'queries_used': search_queries,
            'findings': []
        }
        
        # Identify URLs to scrape
        urls_to_scrape = [r for r in ranked_results 
                         if r['url'] not in self.scraped_urls 
                         and r.get('scrape_decision', {}).get('should_scrape', False)][:10]
        
        if urls_to_scrape:
            # Extract content from URLs
            contents = await self.batch_extract_content(
                [r['url'] for r in urls_to_scrape]
            )
            
            for result in urls_to_scrape:
                url = result['url']
                content = contents.get(url, '')
                if content:
                    self.scraped_urls.add(url)
                    finding = {
                        'source': url,
                        'content': content[:self.get_scrape_limit(
                            result.get('scrape_decision', {}).get('scrape_level', 'MEDIUM')
                        )],
                        'relevance_score': result.get('relevance_score', 0),
                        'is_top_result': False,
                        'scrape_level': result.get('scrape_decision', {}).get('scrape_level', 'MEDIUM')
                    }
                    iteration_data['findings'].append(finding)
                    research_data['final_sources'].append(finding)
            
            self.logger.info(f"Added {len(iteration_data['findings'])} findings from {len(urls_to_scrape)} URLs")
        
        return iteration_data

    async def research(self, query: str) -> str:
        """Main research function that coordinates the entire research process."""
        self.reset_state()
        self.logger.info(f"Starting research: {query}")
        
        research_data = {
            'main_query': query,
            'iterations': [],
            'final_sources': [],
            'report_versions': []  # Track different versions of the report
        }
        
        # Store the latest report structure
        latest_report_structure = ""
        
        while self.research_iterations < self.MAX_ITERATIONS:
            # Get search queries for this iteration
            if self.research_iterations == 0:
                # First iteration: generate initial queries
                search_queries = self.generate_subqueries(query, research_data)
                self.previous_queries.update(search_queries)
            else:
                # For subsequent iterations, we should already have new_queries defined
                if not hasattr(self, '_current_queries') or not self._current_queries:
                    self.logger.warning("No additional queries for this iteration")
                    break
                search_queries = [q for q in self._current_queries if q not in self.previous_queries]
                if not search_queries:
                    self.logger.warning("No new unique queries to process")
                    break
                self.previous_queries.update(search_queries)
            
            self.logger.info(f"Processing {len(search_queries)} queries for iteration {self.research_iterations + 1}")
            
            # Perform research cycle
            iteration_data = await self.perform_research_cycle(search_queries, research_data)
            if iteration_data:
                research_data['iterations'].append(iteration_data)
            
            self.research_iterations += 1
            
            if self.research_iterations >= self.MAX_ITERATIONS:
                break
            
            # Clear current queries after processing
            self._current_queries = []
        
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

        # Generate initial report
        self.logger.info("Generating initial report...")
        initial_report_text, initial_sources_used = await self.generate_report(
            query, research_data, latest_report_structure or """
            # Research Report: {query}
            
            ## Summary
            Present the main findings or lack thereof.
            
            ## Available Information
            Detail any information found, even if limited.
            
            ## Limitations
            Explain why information might be limited or unavailable.
            
            ## Recommendations
            Suggest alternative approaches or queries if needed.
            """
        )
        
        if initial_report_text:
            # Store initial report version
            research_data['report_versions'].append({
                'version': 'initial',
                'report_text': initial_report_text,
                'sources_used': initial_sources_used
            })
            
            # Review the report
            self.logger.info("Reviewing report quality...")
            approved, review_notes, missing_aspects, additional_queries = await self.review_report(
                query, initial_report_text, research_data
            )
            
            # Store review information
            research_data['review_data'] = {
                'approved': approved,
                'review_notes': review_notes,
                'missing_aspects': missing_aspects,
                'suggested_queries': additional_queries
            }
            
            # If not approved and we haven't hit max iterations, do one more research cycle
            if not approved and self.research_iterations < self.MAX_ITERATIONS:
                self.logger.info(f"Report review indicated improvements needed:\n{review_notes}")
                
                if missing_aspects and additional_queries:
                    self.logger.info(f"Missing aspects identified: {', '.join(missing_aspects)}")
                    self.research_iterations += 1
                    
                    # Add new queries to previous_queries
                    if additional_queries:
                        self.logger.info(f"Adding {len(additional_queries)} new search queries")
                        
                        # Store queries for next iteration
                        self._current_queries = additional_queries
                        
                        # Perform additional research cycle with new queries
                        new_iteration_data = await self.perform_research_cycle(
                            additional_queries, 
                            research_data
                        )
                        
                        if new_iteration_data:
                            research_data['iterations'].append(new_iteration_data)
                
                # Generate final report with improved content, including context from initial report and review
                self.logger.info("Generating improved report with additional research...")
                final_report_text, final_sources_used = await self.generate_report(
                    query,
                    {
                        **research_data,
                        'initial_report': initial_report_text,
                        'review_feedback': {
                            'notes': review_notes,
                            'missing_aspects': missing_aspects
                        }
                    },
                    latest_report_structure + """
                    
                    ## Previous Research Context
                    This section addresses gaps identified in the initial research:
                    
                    ### Initial Findings
                    Summary of initial research findings
                    
                    ### Identified Gaps
                    List of gaps identified in the review
                    
                    ### Additional Research
                    New findings from follow-up research
                    
                    ### Synthesis
                    Integration of initial and new findings
                    """
                )
                
                if final_report_text:
                    # Store final report version
                    research_data['report_versions'].append({
                        'version': 'final',
                        'report_text': final_report_text,
                        'sources_used': final_sources_used
                    })
                    report_text = final_report_text
                    sources_used = final_sources_used
                else:
                    # Fallback to initial report if final generation fails
                    self.logger.warning("Failed to generate improved report, using initial version")
                    report_text = initial_report_text
                    sources_used = initial_sources_used
            else:
                if approved:
                    self.logger.info("Initial report approved by review process")
                else:
                    self.logger.info("Max iterations reached, proceeding with initial report")
                report_text = initial_report_text
                sources_used = initial_sources_used
            
            # Save the final report
            if report_text:
                report_file = self.save_report_streaming(
                    query, report_text, sources_used
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