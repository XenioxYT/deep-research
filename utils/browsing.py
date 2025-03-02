import re
import asyncio
import logging
import colorlog
from urllib.parse import urlparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

class BrowserManager:
    def __init__(self):
        """Initialize browser manager with logging."""
        self.setup_logging()
        self.transcript_formatter = TextFormatter()

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
        
        logger = colorlog.getLogger('browser_manager')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        self.logger = logger

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

    def get_youtube_video_id(self, url: str) -> str:
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

    async def get_youtube_transcript(self, url: str) -> str:
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
                    await asyncio.sleep(0.5)  # Wait before retry
                return ""
            finally:
                if page:
                    try:
                        await page.close()
                    except:
                        pass
        
        return ""

    async def batch_extract_content(self, urls: list[str], max_concurrent: int = 8) -> dict[str, str]:
        """Extract content from multiple URLs in parallel with enhanced concurrency."""
        # Limit the number of URLs to process
        
        self.logger.info(f"Extracting content from {len(urls)} URLs")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create a pool of browser pages
        page_pool = []
        for _ in range(max_concurrent):
            page = await self.context.new_page()
            page_pool.append(page)
        
        async def extract_with_page_pool(url: str) -> tuple[str, str]:
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

    async def search_duckduckgo(self, query: str, num_results: int = 10) -> list[dict]:
        """
        Search DuckDuckGo HTML version and extract results.
        
        Args:
            query: The search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of results in Google CSE format
        """
        self.logger.info(f"Searching DuckDuckGo HTML for: {query}")
        results = []
        page = None
        
        try:
            # Format the query for URL
            encoded_query = query.replace(' ', '+')
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            page = await self.context.new_page()
            await page.goto(url, wait_until='domcontentloaded', timeout=15000)
            
            # Wait for search results to load
            await page.wait_for_selector('.result.results_links', timeout=10000)
            
            # Extract results
            result_elements = await page.query_selector_all('.result.results_links')
            
            for element in result_elements:
                if len(results) >= num_results:
                    break
                    
                try:
                    # Check if result is an ad
                    ad_badge = await element.query_selector('.badge--ad')
                    if ad_badge:
                        self.logger.debug("Skipping ad result")
                        continue
                    
                    # Extract title
                    title_element = await element.query_selector('.result__title .result__a')
                    title = await title_element.inner_text() if title_element else "No title"
                    
                    # Extract display link first as it's the clean URL
                    display_link_element = await element.query_selector('.result__url')
                    display_link = await display_link_element.inner_text() if display_link_element else ""
                    display_link = display_link.strip()
                    
                    # Use display link as the main URL, adding https:// if needed
                    url = f"https://{display_link}" if display_link and not display_link.startswith(('http://', 'https://')) else display_link
                    
                    # Extract description/snippet
                    snippet_element = await element.query_selector('.result__snippet')
                    snippet = await snippet_element.inner_text() if snippet_element else ""
                    
                    # Clean up the text
                    title = title.strip()
                    snippet = snippet.strip()
                    
                    # Format in Google CSE style
                    result = {
                        'link': url,
                        'title': title,
                        'snippet': snippet,
                        'displayLink': display_link
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting search result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo HTML search error: {str(e)}")
            return []
            
        finally:
            if page:
                await page.close()
                
    async def search_duckduckgo_regular(self, query: str, num_results: int = 10) -> list[dict]:
        """
        Search regular DuckDuckGo (not HTML-only version) and extract results.
        
        Args:
            query: The search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of results in Google CSE format
        """
        self.logger.info(f"Searching regular DuckDuckGo for: {query}")
        results = []
        page = None
        
        try:
            # Format the query for URL
            encoded_query = query.replace(' ', '+')
            url = f"https://duckduckgo.com/?q={encoded_query}"
            
            page = await self.context.new_page()
            await page.goto(url, wait_until='domcontentloaded', timeout=15000)
            
            # Wait for search results to load - using the article selector from regular DDG
            await page.wait_for_selector('article[data-testid="result"]', timeout=10000)
            
            # Extract results
            result_elements = await page.query_selector_all('article[data-testid="result"]')
            
            for element in result_elements:
                if len(results) >= num_results:
                    break
                    
                try:
                    # Check if result is an ad
                    ad_badge = await element.query_selector('.badge--ad')
                    if ad_badge:
                        self.logger.debug("Skipping ad result")
                        continue
                    
                    # Extract title
                    title_element = await element.query_selector('h2 a span')
                    title = await title_element.inner_text() if title_element else "No title"
                    
                    # Extract URL
                    url_element = await element.query_selector('h2 a')
                    url = await url_element.get_attribute('href') if url_element else ""
                    
                    # Extract display link
                    display_link_element = await element.query_selector('.veU5I0hFkgFGOPhX2RBE span')
                    display_link = await display_link_element.inner_text() if display_link_element else ""
                    
                    # Extract description/snippet
                    snippet_element = await element.query_selector('.kY2IgmnCmOGjharHErah')
                    snippet = await snippet_element.inner_text() if snippet_element else ""
                    
                    # Clean up the text
                    title = title.strip()
                    snippet = snippet.strip()
                    display_link = display_link.strip()
                    
                    # Format in Google CSE style
                    result = {
                        'link': url,
                        'title': title,
                        'snippet': snippet,
                        'displayLink': display_link
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting regular DuckDuckGo search result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Regular DuckDuckGo search error: {str(e)}")
            return []
            
        finally:
            if page:
                await page.close() 