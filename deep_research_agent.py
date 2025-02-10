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
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

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
            'gemini-2.0-pro-exp-02-05',
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
        
        # Initialize YouTube transcript formatter
        self.transcript_formatter = TextFormatter()

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
        """Check if URL is a YouTube video (not channel/user/playlist)."""
        if not any(yt in urlparse(url).netloc for yt in ['youtube.com', 'youtu.be']):
            return False
            
        # Check if it's a video URL by looking for specific patterns
        video_patterns = [
            r'youtube\.com/watch\?v=',  # Standard video URL
            r'youtu\.be/',             # Short URL
            r'youtube\.com/embed/',     # Embedded video
            r'youtube\.com/v/',         # Old style video URL
            r'youtube\.com/shorts/'     # YouTube shorts
        ]
        
        # Return True only if it matches a video pattern
        return any(re.search(pattern, url) for pattern in video_patterns)

    def rewrite_url(self, url: str) -> str:
        """Rewrite URLs based on defined rules to use alternative frontends."""
        parsed = urlparse(url)
        
        # Dictionary of domain rewrites for better scraping
        rewrites = {
            # Social media
            'twitter.com': 'nitter.net',
            'x.com': 'nitter.net',
            'reddit.com': 'redlib.kylrth.com',
            'instagram.com': 'imginn.com',
            
            # News and articles
            'medium.com': 'scribe.rip',
            'bloomberg.com': 'archive.ph',
            'wsj.com': 'archive.ph',
            'nytimes.com': 'archive.ph',
            'ft.com': 'archive.ph',
        }
        
        # Extract domain without subdomain
        domain = '.'.join(parsed.netloc.split('.')[-2:])
        
        # Check if domain needs rewriting
        if domain in rewrites:
            new_domain = rewrites[domain]
            
            # Special handling for archive.ph (needs full URL encoded)
            if new_domain == 'archive.ph':
                return f'https://archive.ph/{url}'
            
            # For all other rewrites, maintain the path and query
            new_url = url.replace(parsed.netloc, new_domain)
            
            # Log the rewrite
            self.logger.info(f"Rewriting URL: {url} -> {new_url}")
            return new_url
            
        return url

    def should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped."""
        return (
            url in self.blacklisted_urls or
            any(ext in url for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx'])
        )

    def generate_subqueries(self, main_query: str, research_state: Optional[Dict] = None) -> List[str]:
        """Generate sub-queries using AI to explore different aspects of the main query."""
        self.logger.info("Analyzing query and generating search queries...")
        
        MAX_QUERIES = 15  # Maximum number of queries to return
        
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
           - Exact phrase matches in quotes
           - Site-specific searches (e.g., site:edu, site:gov)
           - Date-range specific queries when relevant
           - Include synonyms and related terms
           - Combine key concepts in different ways
           - Use both broad and specific queries
           - Target authoritative sources

        4. Response Format:
        TYPE: [SIMPLE/COMPLEX]
        REASON: [One clear sentence explaining the classification]
        QUERIES:
        [If SIMPLE: Only output the original query
        If COMPLEX: Generate up to {MAX_QUERIES} search queries that:
        - Start each line with a number
        - Include exact phrases in quotes when relevant
        - Use site: operators for authoritative sources
        - Combine terms to target specific aspects
        - Avoid redundant queries
        - Focus on the most important aspects first
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
        10. future "workplace trends" expert predictions
        11. "remote work" cost savings analysis
        12. site:linkedin.com remote work success stories
        13. "distributed teams" management strategies
        14. remote work technology infrastructure requirements
        15. "work from home" impact on cities"""

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
                # Check if it's a YouTube URL
                if self.is_youtube_url(url):
                    transcript = await self.get_youtube_transcript(url)
                    if transcript:
                        return url, transcript
                    return url, ""

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
           - LOW: 300 chars - For basic/overview content (default)
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
        2. For queries needing more depth/verification:
           - On iteration 0-2: Say YES if significant information is missing
           - On iteration 3: Only say YES if crucial information is missing
           - On iteration 4+: Strongly lean towards NO unless absolutely critical information is missing
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
        SEARCH_QUERIES: [List complete search queries, one per line, max 10. Search formatting and quotes are allowed. These queries should be specific to the information you are looking for.]
        SCRAPE_NEXT: [List URLs to scrape in next iteration, one per line, in format: URL | LOW/MEDIUM/HIGH]
        REPORT_STRUCTURE: [A complete, customized report structure and guidelines based on the query type and findings. This should include:
        1. Title of the report - Transform the query into a professional academic/research title:
           - Capitalize important words
           - Remove question marks and informal language
           - Add relevant context from findings
           - Format as a research paper title
           Examples:
           Query: "how does ai impact healthcare" → "The Impact of Artificial Intelligence on Modern Healthcare Systems"
           Query: "what are the effects of climate change" → "Analysis of Climate Change Effects: Global Environmental Impact Assessment"
           Query: "compare tesla and ford" → "Comparative Analysis of Tesla and Ford: Manufacturing, Innovation, and Market Position"
        2. Required sections and their order (Do not include an introduction section)
        3. What to include in each section
        4. Specific formatting guidelines
        5. Try to include a table or information where relevant
        6. Any special considerations for this topic
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
                # {A professional title based on the query, following academic style}
                
                ## Direct Answer
                {Provide the direct, factual answer to the query}
                
                ## Explanation
                {Provide a clear explanation of the answer, including any relevant context or background information}
                
                ## Additional Context
                {Include any relevant additional information, examples, or related concepts that help understand the answer better}
                
                ## Opinion
                {A brief, 1-2 sentence opinion on the topic and the sources used}
                """
            
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
                Using the following information:
                {json.dumps(report_context, indent=2)}
                
                Follow this custom report structure and guidelines:
                {report_structure}
                
                For simple queries (mathematical, factual, or definitional):
                - Keep the title professional but concise (e.g., "Mathematical Analysis: Sum of 2 and 2")
                - Use # for the main title
                - Use ## for main sections
                - Use ### for subsections if needed
                - Provide a clear, direct answer
                - Include a brief explanation of the concept if relevant
                - Keep additional context minimal and focused
                
                For complex queries:
                - Create a detailed, academic-style title with # heading
                - Use ## for main sections
                - Use ### for subsections
                - Use #### for detailed subsection breakdowns where needed
                - Include comprehensive analysis of all relevant information
                - Address any contradictions or nuances in the sources
                - Provide thorough explanations and context
                
                General Guidelines:
                - The report should be detailed and include all relevant information from sources
                - Always use proper heading hierarchy (# → ## → ### → ####)
                - Use **bold** for emphasis on key points
                - Format numbers naturally with proper thousands separators
                - Use [1][2][3] format for references, not [1, 2, 3]
                - Mention when using knowledge beyond the sources and note potential for hallucination
                - Use LaTeX for equations ($$ for display math, $ for inline math)
                - Use LaTeX for superscript/subscript (^{2}, _{2}). DO NOT use HTML formatting, like <sup> or <sub>.
                
                Start the report immediately without any additional formatting or preamble.
                Format in clean Markdown without code blocks (unless showing code snippets).
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

    async def research(self, query: str) -> str:
        """Main research function that coordinates the entire research process."""
        self.reset_state()
        self.logger.info(f"Starting research: {query}")
        
        research_data = {
            'main_query': query,
            'iterations': [],
            'final_sources': []
        }
        
        # Store the latest report structure
        latest_report_structure = ""
        
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
                    rewritten_url = self.rewrite_url(url)
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
                
                # Parallel content extraction
                contents = await self.batch_extract_content(list(urls_to_extract))
                
                # Process extracted content in parallel
                async def process_content(result):
                    url = result['url']
                    content = contents.get(url, '')
                    if not content:
                        return None
                        
                    self.scraped_urls.add(url)
                    rewritten_url = self.rewrite_url(url)
                    
                    # Remove individual content storage logs
                    if url == top_url:
                        content_to_store = content[:10000]
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
            need_more_research, explanation, new_queries, report_structure = self.analyze_research_state(
                query, analysis_context
            )
            
            if report_structure:
                latest_report_structure = report_structure
                self.logger.info("Updated report structure based on latest analysis")
            
            if need_more_research and self.research_iterations < self.MAX_ITERATIONS - 1:
                self.logger.info(f"Continuing research - Iteration {self.research_iterations + 1}")
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
            query, research_data, latest_report_structure or """
            # {A professional title based on the query, following academic style}
            
            ## Direct Answer
            {Provide the direct, factual answer to the query}
            
            ## Explanation
            {Provide a clear explanation of the answer, including any relevant context or background information}
            
            ## Additional Context
            {Include any relevant additional information, examples, or related concepts that help understand the answer better}
            """
        )
        
        if report_generator:
            report_file = self.save_report_streaming(
                query, report_generator, sources_used
            )
            if report_file:
                self.logger.info(f"Report has been generated and saved to: {report_file}")
                return f"Report has been generated and saved to: {report_file}"
        
        return "Error: Failed to generate report. Please try again."

    def get_youtube_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        # First check if it's actually a video URL
        if not self.is_youtube_url(url):
            return None
            
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([^&\n?#]+)',
            r'youtube\.com\/shorts\/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    async def get_youtube_transcript(self, url: str) -> Optional[str]:
        """Get transcript from YouTube video."""
        try:
            video_id = self.get_youtube_video_id(url)
            if not video_id:
                self.logger.warning(f"Could not extract video ID from URL: {url}")
                return None

            # Get available transcripts
            transcript_list = await asyncio.to_thread(
                YouTubeTranscriptApi.list_transcripts,
                video_id
            )

            # Try to get English transcript (manual or auto-generated)
            try:
                transcript = await asyncio.to_thread(
                    transcript_list.find_transcript,
                    ['en']
                )
            except:
                # If no English transcript, try to get any transcript and translate it
                try:
                    transcript = await asyncio.to_thread(
                        transcript_list.find_manually_created_transcript
                    )
                    transcript = await asyncio.to_thread(
                        transcript.translate,
                        'en'
                    )
                except:
                    # If no manual transcript, try auto-generated
                    try:
                        transcript = await asyncio.to_thread(
                            transcript_list.find_generated_transcript
                        )
                        if transcript.language_code != 'en':
                            transcript = await asyncio.to_thread(
                                transcript.translate,
                                'en'
                            )
                    except Exception as e:
                        self.logger.warning(f"No transcript available for video {video_id}: {str(e)}")
                        return None

            # Get transcript data and format it
            transcript_data = await asyncio.to_thread(transcript.fetch)
            formatted_transcript = self.transcript_formatter.format_transcript(transcript_data)
            
            return formatted_transcript

        except Exception as e:
            self.logger.warning(f"Error getting transcript for {url}: {str(e)}")
            return None

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