# app.py
import os
import random
import json
import time
import logging
import asyncio
import uuid
import requests
from typing import List, Dict, Any, Optional, Annotated
from datetime import datetime, timedelta
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import aiohttp
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os.path
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# For Semantic Kernel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.agents import ChatCompletionAgent, SequentialOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments
# Add this to the imports
import random
import time
from collections import deque


# Environment variables - in production, set these as app settings in Azure App Service
AZURE_OPENAI_ENDPOINT = "https://prodhubfinnew-openai-97de.openai.azure.com/"
AZURE_OPENAI_API_KEY = "97fa8c02f9e64e8ea5434987b11fe6f4"  # In production, use environment variables
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
CHAT_DEPLOYMENT = "gpt-4.1"  # Replace with your actual deployment name
BLOCKED_DOMAINS = set()
FAILED_URLS = set()  # URLs that have permanently failed (e.g., 404)
URL_CACHE = {}       # Cache of successfully fetched content
MAX_CACHE_SIZE = 1000  # Maximum number of cached responses
URL_RETRY_LIMITS = {}  # Track retry attempts per domain

# Initialize the FastAPI app
app = FastAPI(title="Competitor Intelligence & Market Trends Agent", 
              description="AI agent system for competitor analysis and market trends monitoring")

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data storage - In production, consider using Azure Blob Storage or other persistent storage
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ================ Models ================

class CompetitorInfo(BaseModel):
    name: str
    url: str
    similarity_score: float
    description: str = ""

class ProductInfo(BaseModel):
    id: str
    name: str
    price: float
    description: str = ""
    url: str
    features: List[str] = []
    last_updated: str

class MarketTrend(BaseModel):
    trend: str
    sources: List[str]
    relevance_score: float
    description: str

class MarketSentiment(BaseModel):
    overall_sentiment: float  # -1.0 to 1.0
    key_points: List[str]
    sources: List[str]

class InsightItem(BaseModel):
    insight: str
    relevance: float
    action_items: List[str]

class CompetitorAnalysisReport(BaseModel):
    timestamp: str
    company_url: str
    company_name: str
    competitors: List[CompetitorInfo]
    products: Dict[str, List[ProductInfo]]
    market_trends: List[MarketTrend]
    market_sentiment: MarketSentiment
    insights: List[InsightItem]
    recommendations: List[str]

class CompetitorRequest(BaseModel):
    url: HttpUrl
    depth: int = 3  # Depth of analysis (1-5)

class CompetitorResponse(BaseModel):
    task_id: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: Optional[CompetitorAnalysisReport] = None

# ================ DuckDuckGo Search Manager ================
# ================ Proxy Management ================

class ProxyManager:
    """Manages rotating proxies for requests"""
    
    def __init__(self, proxies=None):
        """
        Initialize with a list of proxies.
        
        Args:
            proxies (list): List of proxy strings in format "http://user:pass@host:port"
                           If None, no proxies will be used
        """
        self.proxies = proxies or []
        self.current_index = 0
        self.last_rotation = time.time()
        self.rotation_interval = 10  # seconds between proxy rotations
        
    def get_proxy(self):
        """Get the current proxy or None if no proxies are available"""
        if not self.proxies:
            return None
            
        # Check if it's time to rotate
        now = time.time()
        if now - self.last_rotation >= self.rotation_interval:
            self.current_index = (self.current_index + 1) % len(self.proxies)
            self.last_rotation = now
            
        return self.proxies[self.current_index]
        
    def mark_proxy_failed(self):
        """Mark the current proxy as failed and rotate to the next one"""
        if not self.proxies:
            return
            
        # Remove the failed proxy if we have others
        if len(self.proxies) > 1:
            self.proxies.pop(self.current_index)
            self.current_index = self.current_index % len(self.proxies)
        
        self.last_rotation = time.time()

# Initialize with empty proxy list - would be populated from config or environment in production
proxy_manager = ProxyManager()

# ================ Rate Limiting ================

class RateLimiter:
    """Enforces rate limits for different domains"""
    
    def __init__(self):
        self.request_times = {}
        self.max_requests_per_second = 10  # Default limit
        self.domain_limits = {
            "duckduckgo.com": 2,  # Very conservative rate limit for DuckDuckGo
            "html.duckduckgo.com": 1,
            "lite.duckduckgo.com": 1
        }
    
    def wait_if_needed(self, domain):
        """Wait if necessary to respect rate limits for the domain"""
        # Get the appropriate rate limit for this domain
        rate_limit = self.domain_limits.get(domain, self.max_requests_per_second)
        
        # Get the request history for this domain
        if domain not in self.request_times:
            self.request_times[domain] = deque(maxlen=rate_limit)
        
        # Calculate if we need to wait
        now = time.time()
        if len(self.request_times[domain]) >= rate_limit:
            oldest_request = self.request_times[domain][0]
            time_diff = now - oldest_request
            
            # If we've made too many requests within 1 second
            if time_diff < 1.0:
                # Sleep until the oldest request is 1 second old
                sleep_time = 1.0 - time_diff + random.uniform(0.1, 0.3)  # Add jitter
                time.sleep(sleep_time)
        
        # Record this request time
        self.request_times[domain].append(time.time())

# Initialize the rate limiter
rate_limiter = RateLimiter()
class DuckDuckGoSearchManager:
    """
    A class to perform various types of web searches using DuckDuckGo with proxy rotation.
    """

    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
        ]
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 5  # Minimum seconds between requests to avoid rate limiting
        self.max_retries = 1
        
    def _get_random_user_agent(self):
        return random.choice(self.user_agents)
    
    def _prepare_request(self, url, params=None):
        """Prepare a request with proper headers, delays, and proxy rotation"""
        # Apply rate limiting based on domain
        domain = urlparse(url).netloc
        rate_limiter.wait_if_needed(domain)
        
        # Get a proxy if available
        proxy = proxy_manager.get_proxy()
        proxies = {"http": proxy, "https": proxy} if proxy else None
        
        # Prepare headers
        headers = {
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'Referer': 'https://duckduckgo.com/',
        }
        
        # Add a delay between requests
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed + random.uniform(0.1, 0.5))  # Add jitter
        self.last_request_time = time.time()
        
        return headers, proxies

    def text_search(self, query, num_results=3) -> list:
        """
        Performs a DuckDuckGo text search and returns a list of URLs.
        """
        logger.info(f"Performing DuckDuckGo text search for: {query}")
        
        # Try different DuckDuckGo endpoints
        endpoints = [
            "https://html.duckduckgo.com/html/",
            "https://lite.duckduckgo.com/lite/",
            "https://duckduckgo.com/html"
        ]
        
        for endpoint in endpoints:
            for attempt in range(self.max_retries):
                try:
                    # Prepare the request
                    headers, proxies = self._prepare_request(endpoint)
                    
                    # First, visit the DuckDuckGo homepage to get cookies
                    self.session.get('https://duckduckgo.com/', headers=headers, proxies=proxies, timeout=10)
                    
                    # Then make the search request
                    params = {'q': query}
                    response = self.session.get(
                        endpoint, 
                        headers=headers, 
                        params=params, 
                        proxies=proxies,
                        timeout=10
                    )
                    
                    logger.info(f"DuckDuckGo search status code: {response.status_code} for endpoint {endpoint}")
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        results = []
                        
                        # Handle different HTML structures based on the endpoint
                        if 'lite.duckduckgo.com' in endpoint:
                            # Parse lite version results
                            links = soup.select('a[href]')
                            for link in links:
                                href = link.get('href')
                                if href and not href.startswith('/') and 'duckduckgo.com' not in href:
                                    results.append(href)
                                    if len(results) >= num_results:
                                        return results
                        else:
                            # Parse standard version results - handle different selectors
                            selectors = [
                                '.result__url', 
                                '.result__a', 
                                '.result__title a', 
                                '.links_main a', 
                                '.web-result a'
                            ]
                            
                            for selector in selectors:
                                for result in soup.select(selector):
                                    href = None
                                    if result.has_attr('href'):
                                        href = result['href']
                                    elif result.has_attr('data-href'):
                                        href = result['data-href']
                                        
                                    if href and not href.startswith('/') and 'duckduckgo.com' not in href:
                                        results.append(href)
                                
                                if results:  # If we found results with this selector, stop trying others
                                    break
                            
                            # Limit to requested number
                            results = results[:num_results]
                            
                            if results:
                                return results
                    
                    elif response.status_code == 202:
                        logger.warning(f"DuckDuckGo search returned 202 status - retry attempt {attempt+1}")
                        # Try with a different proxy if available
                        proxy_manager.mark_proxy_failed()
                        time.sleep(2 + random.uniform(0.5, 1.5))  # Increasing delay between retries
                    
                    elif response.status_code in (403, 429):
                        logger.warning(f"DuckDuckGo rate limited or blocked - status {response.status_code}")
                        # Definitely try a different proxy
                        proxy_manager.mark_proxy_failed()
                        time.sleep(5 + random.uniform(1, 3))  # Longer delay for rate limiting
                        
                    else:
                        logger.warning(f"Unexpected status code {response.status_code} from DuckDuckGo")
                        time.sleep(1 + random.uniform(0.5, 1.0))
                
                except Exception as e:
                    logger.error(f"Error during DuckDuckGo text search attempt {attempt+1}: {str(e)}")
                    time.sleep(2)
        
        # If all attempts with all endpoints failed, return empty list
        logger.error(f"All DuckDuckGo search attempts failed for query: {query}")
        return []

    def news_search(self, query, num_results=3) -> list:
        """
        Performs a DuckDuckGo news search and returns a list of news article URLs.

        Parameters:
        - query (str): The search query string for finding relevant news articles.
        - num_results (int): The maximum number of news article URLs to return. Defaults to 3.

        Returns:
        - list of str: A list containing the URLs of the news articles. Each URL in the list corresponds to a news article that matches the search query.
        """
        try:
            self._delay_if_needed()
            
            # Set headers with a random user agent
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Referer': 'https://duckduckgo.com/',
            }
            
            # Add news filter to the query
            modified_query = f"{query} site:news.google.com OR site:reuters.com OR site:bloomberg.com OR site:bbc.com OR site:nytimes.com"
            
            # First, visit the DuckDuckGo homepage to get cookies
            self.session.get('https://duckduckgo.com/', headers=headers)
            
            # Then make the search request
            search_url = f"https://html.duckduckgo.com/html/"
            params = {'q': modified_query}
            response = self.session.get(search_url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo news search failed with status code: {response.status_code}")
                # Try the alternative lite version
                search_url = f"https://lite.duckduckgo.com/lite/"
                response = self.session.get(search_url, headers=headers, params=params)
                
                if response.status_code != 200:
                    logger.warning(f"DuckDuckGo lite news search also failed with status code: {response.status_code}")
                    return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract links from search results
            if 'lite.duckduckgo.com' in response.url:
                # Parse lite version results
                links = soup.select('a[href]')
                for link in links:
                    href = link.get('href')
                    if href and not href.startswith('/') and 'duckduckgo.com' not in href:
                        results.append(href)
                        if len(results) >= num_results:
                            break
            else:
                # Parse standard version results
                for result in soup.select('.result__url, .result__a'):
                    href = result.get('href')
                    if href and not href.startswith('/') and 'duckduckgo.com' not in href:
                        # For result__url, href is direct. For result__a, extract from data-href
                        if 'data-href' in result.attrs:
                            href = result['data-href']
                        results.append(href)
                        if len(results) >= num_results:
                            break
            
            return results
        except Exception as e:
            logger.error(f"Error during DuckDuckGo news search: {str(e)}")
            return []
            
    def images_search(self, query, num_results=3) -> list:
        """
        Performs a DuckDuckGo image search and returns a list of dictionaries, each containing URLs for an image and its thumbnail.

        Parameters:
        - query (str): The search query for the image search.
        - num_results (int): The maximum number of image results to return. Defaults to 3.

        Returns:
        - list of dict: A list where each element is a dictionary with two keys:
            'image': URL of the actual image,
            'thumbnail': URL of the thumbnail of the image.
        """
        try:
            self._delay_if_needed()
            
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Referer': 'https://duckduckgo.com/',
                'X-Requested-With': 'XMLHttpRequest',
                'DNT': '1',
            }
            
            # Visit the DuckDuckGo homepage to get a VQD token
            home_response = self.session.get('https://duckduckgo.com/', headers=headers)
            home_content = home_response.text
            
            # Extract VQD token
            vqd_match = re.search(r'vqd=\'([-0-9]+)\'', home_content)
            if not vqd_match:
                logger.error("Could not extract VQD token for image search")
                return []
                
            vqd = vqd_match.group(1)
            
            # Make the image search request
            images_url = "https://duckduckgo.com/i.js"
            params = {
                'q': query,
                'o': 'json',
                'vqd': vqd,
                'f': ',,,',
                'p': '1'
            }
            
            response = self.session.get(images_url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo images search failed with status code: {response.status_code}")
                return []
                
            try:
                results = response.json().get('results', [])
                image_info = []
                
                for result in results[:num_results]:
                    if 'image' in result and 'thumbnail' in result:
                        image_info.append({
                            'image': result['image'],
                            'thumbnail': result['thumbnail']
                        })
                
                return image_info
            except ValueError as e:
                logger.error(f"Error parsing image search results: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error during DuckDuckGo images search: {str(e)}")
            return []

    def videos_search(self, query, num_results=3):
        """
        Performs a DuckDuckGo videos search and returns a list of dictionaries, each containing the title and content URL of a video.

        Parameters:
        - query (str): The search query string for finding relevant video results.
        - num_results (int): The maximum number of video results to return. Defaults to 3.

        Returns:
        - list of dict: A list where each dictionary contains 'title' and 'content' keys.
          'title' is the title of the video, and 'content' is the URL of the video.
        """
        try:
            self._delay_if_needed()
            
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Referer': 'https://duckduckgo.com/',
                'X-Requested-With': 'XMLHttpRequest',
                'DNT': '1',
            }
            
            # Visit the DuckDuckGo homepage to get a VQD token
            home_response = self.session.get('https://duckduckgo.com/', headers=headers)
            home_content = home_response.text
            
            # Extract VQD token
            vqd_match = re.search(r'vqd=\'([-0-9]+)\'', home_content)
            if not vqd_match:
                logger.error("Could not extract VQD token for video search")
                return []
                
            vqd = vqd_match.group(1)
            
            # Make the video search request
            videos_url = "https://duckduckgo.com/v.js"
            params = {
                'q': query,
                'o': 'json',
                'vqd': vqd,
                'f': ',,,',
                'p': '1'
            }
            
            response = self.session.get(videos_url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo videos search failed with status code: {response.status_code}")
                return []
                
            try:
                results = response.json().get('results', [])
                video_info = []
                
                for result in results[:num_results]:
                    if 'title' in result and 'content' in result:
                        video_info.append({
                            'title': result['title'],
                            'content': result['content']
                        })
                
                return video_info
            except ValueError as e:
                logger.error(f"Error parsing video search results: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error during DuckDuckGo videos search: {str(e)}")
            return []

    def maps_search(self, query, place, num_results=3):
        """
        Performs a DuckDuckGo maps search for a specific query and place, returning a list of relevant location details.

        Parameters:
        - query (str): The search query string for finding relevant map results.
        - place (str): The geographical area or location to focus the search on.
        - num_results (int): The maximum number of results to return. Defaults to 3.

        Returns:
        - list of dict: A list where each dictionary contains details about a location
        """
        try:
            # This function relies on a specific API that may not be easily accessible
            # For now, we'll return an empty result and log a warning
            logger.warning("DuckDuckGo maps search is not fully implemented in this version")
            return []
        except Exception as e:
            logger.error(f"Error during DuckDuckGo maps search: {str(e)}")
            return []

# Initialize DuckDuckGo search manager
ddg_search = DuckDuckGoSearchManager()

# ================ Web Content Scraper ================

class WebContentScraper:
    """Class to scrape content from websites"""
    
    def __init__(self, user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"):
        self.headers = {"User-Agent": user_agent}
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
        ]
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 2  # Minimum seconds between requests to avoid rate limiting

    def _get_random_user_agent(self):
        return random.choice(self.user_agents)
        
    def _delay_if_needed(self):
        """Ensures a minimum delay between requests to avoid rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _fetch_page_content(self, url):
        """Fetches the content of a web page from a given URL."""
        try:
            self._delay_if_needed()
            headers = {"User-Agent": self._get_random_user_agent()}
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()  # Raises HTTPError for bad requests
            return response.content
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return None
        except Exception as err:
            logger.error(f"Error occurred: {err}")
            return None

    def _parse_web_content(self, content):
        """Parses HTML content and extracts text from it."""
        try:
            soup = BeautifulSoup(content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text from all paragraph elements
            paragraphs = soup.find_all("p")
            paragraph_text = "\n".join(paragraph.get_text() for paragraph in paragraphs)
            
            # If paragraphs are empty or too short, try getting all text
            if len(paragraph_text) < 500:
                # Get text from the body
                body_text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
                
                # Choose the longer content
                if len(body_text) > len(paragraph_text):
                    return body_text
                    
            return paragraph_text
        except Exception as e:
            logger.error(f"Failed to parse the content: {e}")
            return None

    def scrape_website(self, url):
        """Scrapes the content from a given website URL."""
        logger.debug(f"Scraping URL: {url}")
        page_content = self._fetch_page_content(url)
        if page_content:
            parsed_content = self._parse_web_content(page_content)
            if parsed_content:
                return {"url": url, "content": parsed_content}
            else:
                return {"url": url, "error": "Failed to parse content"}
        return {"url": url, "error": "Failed to fetch page content"}

    def scrape_multiple_websites(self, urls):
        """Scrapes the content from multiple websites."""
        try:
            return json.dumps([self.scrape_website(url) for url in urls], indent=2)
        except Exception as e:
            logger.error(f"Error during scraping multiple websites: {e}")
            return json.dumps({"error": str(e)})

# Initialize the web content scraper
scraper = WebContentScraper()

# ================ Helper Functions ================
def save_task_result(task_id: str, result: CompetitorAnalysisReport):
    """Save task result to disk for persistence"""
    company_domain = extract_domain(result.company_url)
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Create a consistent filename format
    result_filename = f"{company_domain}_{timestamp}.json"
    result_path = os.path.join(DATA_DIR, result_filename)
    
    # Create a metadata file that maps task_id to the actual report filename
    metadata_path = os.path.join(DATA_DIR, f"task_{task_id}_metadata.json")
    
    try:
        # Save the actual report
        with open(result_path, 'w') as f:
            f.write(result.model_dump_json(indent=2))
        
        # Save metadata linking task_id to the report filename
        with open(metadata_path, 'w') as f:
            metadata = {
                "task_id": task_id,
                "company_name": result.company_name,
                "company_domain": company_domain,
                "report_filename": result_filename,
                "timestamp": timestamp
            }
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved report for {result.company_name} with task ID {task_id} to {result_path}")
        return result_filename
    except Exception as e:
        logger.error(f"Error saving task result: {str(e)}")
        return None

def load_task_result(task_id: str) -> Optional[CompetitorAnalysisReport]:
    """Load task result from disk if it exists"""
    # First check if we have metadata for this task
    metadata_path = os.path.join(DATA_DIR, f"task_{task_id}_metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            # Load the metadata to get the actual report filename
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            report_filename = metadata.get("report_filename")
            if report_filename:
                report_path = os.path.join(DATA_DIR, report_filename)
                
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        data = json.load(f)
                        return CompetitorAnalysisReport(**data)
                else:
                    logger.error(f"Report file {report_filename} referenced in metadata doesn't exist")
        except Exception as e:
            logger.error(f"Error loading task metadata or result: {str(e)}")
    
    # Legacy fallback - check for old format task files
    legacy_path = os.path.join(DATA_DIR, f"task_{task_id}.json")
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, 'r') as f:
                data = json.load(f)
                return CompetitorAnalysisReport(**data)
        except Exception as e:
            logger.error(f"Error loading legacy task result: {str(e)}")
    
    return None

def extract_domain(url):
    """Extract the domain name from a URL"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

async def fetch_url(url, timeout=15, max_retries=2, custom_headers=None, cookies=None, respect_blocklist=True):
    """Fetch content from a URL with advanced error handling, caching and blocklist support"""
    # Normalize URL - fix any obvious duplication in path segments
    parsed_url = urlparse(url)
    path_segments = parsed_url.path.split('/')
    normalized_segments = []
    
    # Simple URL normalization to fix duplicated path segments
    for segment in path_segments:
        if segment and (not normalized_segments or segment != normalized_segments[-1]):
            normalized_segments.append(segment)
    
    normalized_path = '/' + '/'.join(filter(None, normalized_segments))
    normalized_url = parsed_url._replace(path=normalized_path).geturl()
    
    # Use normalized URL for all operations
    url = normalized_url
    
    # Check URL cache first for a hit
    if url in URL_CACHE:
        logger.debug(f"Cache hit for {url}")
        return URL_CACHE[url]
    
    # Check if URL is in the permanent failure list
    if url in FAILED_URLS:
        logger.info(f"Skipping {url} as it previously failed with a permanent error")
        return None
    
    # Extract domain to check against blocklist
    domain = parsed_url.netloc
    
    # Check if domain is in the blocklist and we should respect it
    if respect_blocklist and domain in BLOCKED_DOMAINS:
        logger.info(f"Skipping {url} because domain {domain} is in the blocklist")
        return None
    
    # Check domain retry limits
    domain_key = f"domain:{domain}"
    if domain_key in URL_RETRY_LIMITS and URL_RETRY_LIMITS[domain_key] >= 10:
        logger.warning(f"Domain {domain} has exceeded maximum retry attempts, temporarily blocking")
        # Add to blocklist temporarily
        BLOCKED_DOMAINS.add(domain)
        return None
    
    # Rotate user agents to avoid detection
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
    ]
    
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',  # Explicitly exclude 'br' until we're sure it works
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'DNT': '1',
    }
    
    # Update headers with custom headers if provided
    if custom_headers:
        headers.update(custom_headers)
    
    # Configure aiohttp client session with proper settings
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Add a small random delay to simulate more natural browsing
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            async with aiohttp.ClientSession(cookies=cookies, timeout=client_timeout) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    # Handle successful response
                    if response.status == 200:
                        content = await response.text()
                        
                        # Cache the successful result (with size check)
                        if len(URL_CACHE) >= MAX_CACHE_SIZE:
                            # Simple cache eviction - remove a random item
                            URL_CACHE.pop(next(iter(URL_CACHE)))
                        URL_CACHE[url] = content
                        
                        # Reset domain retry counter on success
                        if domain_key in URL_RETRY_LIMITS:
                            URL_RETRY_LIMITS[domain_key] = 0
                            
                        return content
                    
                    # Handle permanent errors - add to permanent failure list
                    elif response.status in [404, 410]:  # Not Found, Gone
                        logger.warning(f"Permanent error for {url}: HTTP {response.status}")
                        FAILED_URLS.add(url)
                        return None
                        
                    # Handle rate limiting or bot detection - use exponential backoff
                    elif response.status in [429, 403, 503]:  # Too Many Requests, Forbidden, Service Unavailable
                        retry_delay = min(2 ** retry_count + random.uniform(0, 1), 60)  # Exponential backoff with max 60 seconds
                        logger.warning(f"Rate limited or blocked on {url}, retrying in {retry_delay} seconds")
                        
                        # Increment domain retry counter
                        URL_RETRY_LIMITS[domain_key] = URL_RETRY_LIMITS.get(domain_key, 0) + 1
                        
                        # For 403, try a different user agent
                        if response.status == 403:
                            headers['User-Agent'] = random.choice(user_agents)
                            
                        await asyncio.sleep(retry_delay)
                    
                    # Handle redirects (should be handled automatically by allow_redirects=True)
                    elif 300 <= response.status < 400:
                        location = response.headers.get('Location')
                        if location:
                            logger.info(f"Redirection from {url} to {location}")
                    
                    # Handle temporary server errors
                    elif 500 <= response.status < 600:
                        logger.warning(f"Server error for {url}: HTTP {response.status}")
                        # Increment retry counter and domain retry counter
                        URL_RETRY_LIMITS[domain_key] = URL_RETRY_LIMITS.get(domain_key, 0) + 1
                        
                        if retry_count + 1 < max_retries:
                            retry_delay = min(2 ** retry_count, 30)
                            await asyncio.sleep(retry_delay)
                        else:
                            return None
                    
                    # Handle other errors
                    else:
                        logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        if retry_count + 1 < max_retries:
                            await asyncio.sleep(2)
                        else:
                            return None
        
        except aiohttp.ClientPayloadError as e:
            # This catches all payload errors including content encoding problems
            logger.error(f"Payload error fetching {url}: {str(e)}")
            # If we encounter an encoding error, try with a different Accept-Encoding header
            if 'br' in str(e).lower() or 'brotli' in str(e).lower():
                headers['Accept-Encoding'] = 'gzip, deflate'  # Ensure 'br' is not included
                logger.info(f"Removed Brotli from accepted encodings for {url}")
            retry_count += 1
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            
            # Increment domain retry counter
            URL_RETRY_LIMITS[domain_key] = URL_RETRY_LIMITS.get(domain_key, 0) + 1
            
            await asyncio.sleep(5)
            retry_count += 1
        except aiohttp.ClientError as e:
            logger.error(f"Client error fetching {url}: {str(e)}")
            
            # Increment domain retry counter
            URL_RETRY_LIMITS[domain_key] = URL_RETRY_LIMITS.get(domain_key, 0) + 1
            
            await asyncio.sleep(3)
            retry_count += 1
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            await asyncio.sleep(3)
            retry_count += 1
    
    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    
    # If we got here, we exceeded retries - increment domain retry counter
    URL_RETRY_LIMITS[domain_key] = URL_RETRY_LIMITS.get(domain_key, 0) + 1
    
    # Try using the web content scraper as a fallback
    try:
        response = scraper.scrape_website(url)
        if "content" in response and response["content"]:
            logger.info(f"Successfully fetched {url} using the web content scraper fallback")
            return response["content"]
    except Exception as e:
        logger.error(f"Fallback web content scraper also failed for {url}: {str(e)}")
    
    return None

async def extract_with_openai(html_content, extraction_goal, openai_client):
    """Extract information from HTML using OpenAI when traditional methods fail"""
    if not html_content:
        return {"error": "No content provided"}
    
    try:
        # Clean HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove scripts, styles, and other non-content elements
        for tag in soup(['script', 'style', 'meta', 'link', 'noscript', 'iframe']):
            tag.extract()
        
        # Extract text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Truncate text to avoid token limits (GPT-4 can handle more but better to be safe)
        text = text[:15000]
        
        # Create the prompt for OpenAI
        prompt = f"""
        You are an expert web data extractor. I'll provide you with text content extracted from a webpage.
        
        Extract the following information: {extraction_goal}
        
        Return ONLY a JSON object with the extracted information. Do not include any explanation or additional text.
        
        Here's the webpage content:
        ---
        {text}
        ---
        """
        
        # Call OpenAI API for extraction
        response = openai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You extract structured data from web content accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Last resort: extract any JSON-like structure
            json_match = re.search(r'(\{[\s\S]*\})', result)
            if json_match:
                return json.loads(json_match.group(1))
            
            return {"error": "Failed to parse OpenAI response", "raw_response": result}
    
    except Exception as e:
        logger.error(f"Error using OpenAI for extraction: {str(e)}")
        return {"error": f"OpenAI extraction failed: {str(e)}"}

def extract_company_info(html_content):
    """Extract company name, description, and other metadata from HTML using OpenAI"""
    if not html_content:
        return {"name": "Unknown", "description": ""}
    
    try:
        # Initialize OpenAI client
        openai_client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        
        # Clean HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove scripts, styles, and other non-content elements
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.extract()
        
        # Extract text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Truncate text to avoid token limits
        text = text[:10000]
        
        # Create the prompt for OpenAI
        prompt = f"""
        You are an expert web data extractor. I'll provide you with text content extracted from a company's webpage.
        
        Extract the following information:
        1. Company name
        2. Company description (a brief summary of what the company does)
        3. Keywords representing the company's industry or focus areas
        4. Social media links (if available)
        
        Return ONLY a JSON object with the extracted information in this format:
        {{
            "name": "Company Name",
            "description": "Brief description of the company",
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "social_links": {{
                "facebook.com": "https://facebook.com/company",
                "twitter.com": "https://twitter.com/company"
            }}
        }}
        
        Here's the webpage content:
        ---
        {text}
        ---
        """
        
        # Call OpenAI API for extraction
        response = openai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You extract structured data from web content accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure we have the required fields
        result["name"] = result.get("name", "Unknown")
        result["description"] = result.get("description", "")
        result["keywords"] = result.get("keywords", [])
        result["social_links"] = result.get("social_links", {})
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting company info with OpenAI: {str(e)}")
        # Fallback to basic extraction using domain name
        try:
            # Try to get title as company name
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.title.string if soup.title else "Unknown"
            return {
                "name": title[:100],  # Limit name length
                "description": "",
                "keywords": [],
                "social_links": {}
            }
        except:
            return {"name": "Unknown", "description": "", "keywords": [], "social_links": {}}

def save_data(data, filename):
    """Save data to a file"""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    return filepath

def load_data(filename):
    """Load data from a file"""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

# Helper for safe JSON parsing with retries
async def safe_parse_json_from_agent(response_text, max_retries=3):
    """Safely parse JSON from agent response with retries"""
    retries = 0
    while retries < max_retries:
        try:
            # Try to extract JSON from the response if it's wrapped in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no code blocks, try to find JSON-like structure
                json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    json_str = response_text.strip()
            
            # Attempt to parse JSON
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (attempt {retries+1}/{max_retries}): {str(e)}")
            retries += 1
            if retries < max_retries:
                # Wait before retrying
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {str(e)}")
            break
    
    # If all retries fail, return a default empty structure
    logger.error(f"Failed to parse JSON after {max_retries} attempts")
    return {"error": "Failed to parse response"}

# Background tasks dictionary to track progress
tasks = {}

# ================ Function to Text Search Using Existing Code ================
async def text_search(query: str, num_results: int = 3) -> list:
    """
    Conducts a general web text search using DuckDuckGo with fallbacks.
    """
    try:
        # Try the improved DuckDuckGo search
        urls = ddg_search.text_search(query, int(num_results))
        
        # If no results, try alternative search or generate using OpenAI
        if not urls:
            logger.warning(f"DuckDuckGo search returned no results for query: {query}")
            
            # Initialize OpenAI client for fallback
            openai_client = AzureOpenAI(
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY
            )
            
            # Use OpenAI to generate information based on the query
            response = openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing information when web search results aren't available."},
                    {"role": "user", "content": f"No web search results were found for the query: '{query}'. Please provide the most relevant information you know about this topic."}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Return a single result with the generated content
            generated_content = response.choices[0].message.content
            return [{"url": "Generated content", "content": generated_content}]
        
        # Scrape content from each URL
        results = []
        for url in urls:
            try:
                # Use scraper to get the content
                result = scraper.scrape_website(url)
                if result and "content" in result and result["content"]:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                # Add error result
                results.append({"url": url, "error": str(e)})
        
        return results
    except Exception as e:
        logger.error(f"Error in text_search: {str(e)}")
        return [{"error": f"Search failed: {str(e)}"}]

# ================ Plugins ================
class WebScraperPlugin:
    """Plugin for web scraping operations using DuckDuckGo and OpenAI"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.ddg_search = ddg_search  
        self.scraper = WebContentScraper()
    
    @kernel_function(description="Scrape a webpage and return its content")
    async def scrape_webpage(self, url: str) -> str:
        """Scrape a webpage and return its content"""
        try:
            result = self.scraper.scrape_website(url)
            if "content" in result and result["content"]:
                # Process the content with OpenAI to extract clean, readable text
                response = self.openai_client.chat.completions.create(
                    model=CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting clean, readable text from web content. Remove noise, ads, and irrelevant information."},
                        {"role": "user", "content": f"Extract the main content from this webpage in a clean, readable format:\n\n{result['content']}"}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                return response.choices[0].message.content[:8000]  # Truncate to avoid token limits
            else:
                # If direct scraping failed, try with our async fetch_url
                content = await fetch_url(url)
                if content:
                    # Process the content with OpenAI
                    response = self.openai_client.chat.completions.create(
                        model=CHAT_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": "You are an expert at extracting clean, readable text from web content. Remove noise, ads, and irrelevant information."},
                            {"role": "user", "content": f"Extract the main content from this webpage in a clean, readable format:\n\n{content[:50000]}"}
                        ],
                        temperature=0.3,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content[:8000]  # Truncate to avoid token limits
                
            return "Failed to scrape the webpage."
        except Exception as e:
            logger.error(f"Error in scrape_webpage: {str(e)}")
            return f"Error scraping webpage: {str(e)}"
    
    @kernel_function(description="Extract all URLs from a webpage")
    async def extract_urls(self, url: str) -> str:
        """Extract all URLs from a webpage"""
        try:
            # First try with the scraper
            html_content = None
            result = self.scraper.scrape_website(url)
            if "content" in result and result["content"]:
                # Convert content back to HTML for URL extraction
                response = self.openai_client.chat.completions.create(
                    model=CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting URLs from web content."},
                        {"role": "user", "content": f"Extract all relevant URLs from this content. Return only a JSON array of URLs.\n\n{result['content']}"}
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                try:
                    result = json.loads(response.choices[0].message.content)
                    if "urls" in result and isinstance(result["urls"], list):
                        return json.dumps(result)
                    else:
                        # If OpenAI extraction failed, fall back to DuckDuckGo
                        domain = extract_domain(url)
                        urls = self.ddg_search.text_search(f"site:{domain}", 5)
                        return json.dumps({"urls": urls})
                except:
                    # If json parsing failed, return an empty result
                    return json.dumps({"urls": []})
            else:
                # If scraper failed, try with async fetch_url
                html_content = await fetch_url(url)
                
            # If we got HTML content, extract URLs
            if html_content:
                soup = BeautifulSoup(html_content, 'html.parser')
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    # Convert relative URLs to absolute
                    if not href.startswith(('http://', 'https://')):
                        base_url = url.rstrip('/')
                        if href.startswith('/'):
                            href = f"{base_url}{href}"
                        else:
                            href = f"{base_url}/{href}"
                    links.append(href)
                
                return json.dumps({"urls": list(set(links))})
            
            # If all attempts failed, use DuckDuckGo search as a last resort
            domain = extract_domain(url)
            urls = self.ddg_search.text_search(f"site:{domain}", 5)
            return json.dumps({"urls": urls})
        except Exception as e:
            logger.error(f"Error in extract_urls: {str(e)}")
            return json.dumps({"urls": [], "error": str(e)})
    
    @kernel_function(description="Extract product information from a webpage")
    async def extract_products(self, url: str) -> str:
        """Extract product information from a webpage using OpenAI"""
        try:
            # First try to get content with the scraper
            scraped_result = self.scraper.scrape_website(url)
            
            html_content = None
            if "content" in scraped_result and scraped_result["content"]:
                html_content = scraped_result["content"]
            else:
                # If scraper failed, try with async fetch_url
                html_content = await fetch_url(url)
            
            if not html_content:
                logger.warning(f"Failed to fetch content from {url}")
                # Try DuckDuckGo search as fallback
                domain = extract_domain(url)
                search_results = self.ddg_search.text_search(f"site:{domain} products", 3)
                
                if not search_results:
                    return json.dumps({"products": []})
                    
                # Try to extract products from search results
                products = []
                for search_url in search_results:
                    search_result = self.scraper.scrape_website(search_url)
                    search_content = search_result.get("content", "") if "content" in search_result else ""
                    
                    if search_content:
                        # Use OpenAI to extract products
                        extracted = await self._extract_products_with_openai(search_content, search_url)
                        if extracted and "products" in extracted and len(extracted["products"]) > 0:
                            products.extend(extracted["products"])
                
                return json.dumps({"products": products})
            
            # Use OpenAI to extract products
            result = await self._extract_products_with_openai(html_content, url)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error in extract_products: {str(e)}")
            return json.dumps({"products": [], "error": str(e)})
    
    async def _extract_products_with_openai(self, html_content, url):
        """Use OpenAI to extract product information from HTML content"""
        try:
            # Create the prompt for OpenAI
            prompt = f"""
            You are an expert web data extractor specializing in product information. I'll provide you with text content extracted from a webpage.
            
            Extract all product information from the page, including:
            - Product name
            - Price (as a numerical value without currency symbols)
            - Description
            - Features
            - URL (use the provided URL if product-specific URLs aren't available)
            
            Return ONLY a JSON object with an array of products in this format:
            {{
                "products": [
                    {{
                        "id": "unique-id",
                        "name": "Product Name",
                        "price": 99.99,
                        "description": "Product description",
                        "url": "https://product-url.com",
                        "features": ["Feature 1", "Feature 2"],
                        "last_updated": "2025-05-19"
                    }}
                ]
            }}
            
            Important guidelines:
            1. Return only products you're confident exist, not guesses.
            2. Use a numerical value for price (e.g., 99.99, not "$99.99").
            3. If no price is found, use 0.0.
            4. Generate a unique ID for each product.
            5. Use today's date (2025-05-19) for the last_updated field.
            6. If no products are found, return an empty products array.
            
            Here's the webpage content:
            ---
            {html_content}
            ---
            
            The URL of this page is: {url}
            """
            
            # Call OpenAI API for extraction
            response = self.openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert web data extractor specializing in product information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Ensure products have all required fields
            products = result.get("products", [])
            for product in products:
                if "id" not in product:
                    product["id"] = str(uuid.uuid4())[:8]
                if "price" not in product:
                    product["price"] = 0.0
                elif isinstance(product["price"], str):
                    # Convert string price to float
                    try:
                        product["price"] = float(product["price"].replace("$", "").replace("€", "").replace("£", "").strip())
                    except:
                        product["price"] = 0.0
                if "features" not in product:
                    product["features"] = []
                if "url" not in product or not product["url"]:
                    product["url"] = url
                if "last_updated" not in product:
                    product["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            
            return {"products": products}
            
        except Exception as e:
            logger.error(f"Error extracting products with OpenAI: {str(e)}")
            return {"products": []}

class WebSearchPlugin:
    """Plugin for web search operations using DuckDuckGo"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.ddg_search = ddg_search  
        self.scraper = WebContentScraper()
    
    @kernel_function(description="Search the web for information about a topic")
    async def search(self, query: str, num_results: int = 5) -> str:
        """Search the web using DuckDuckGo and process results with OpenAI"""
        try:
            # Get URLs from DuckDuckGo
            urls = self.ddg_search.text_search(query, num_results)
            
            if not urls:
                # If DuckDuckGo search fails, use OpenAI to generate a response
                response = self.openai_client.chat.completions.create(
                    model=CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are a search assistant that helps users find information from the web."},
                        {"role": "user", "content": f"Search for information about: {query}. Since I couldn't access search results, provide your best knowledge about this topic."}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                generated_results = [
                    {
                        "title": f"Information about {query}",
                        "snippet": response.choices[0].message.content[:300],
                        "url": "No search results available"
                    }
                ]
                
                return json.dumps({"results": generated_results})
            
            # Scrape content from the URLs
            results = []
            for url in urls:
                scraped_result = self.scraper.scrape_website(url)
                
                if "content" in scraped_result and scraped_result["content"]:
                    content = scraped_result["content"]
                    
                    # Extract title from the URL if not in the content
                    title = extract_domain(url)
                    
                    # Get a snippet using OpenAI
                    snippet_response = self.openai_client.chat.completions.create(
                        model=CHAT_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": "You are a search snippet generator. Create a concise, informative snippet from webpage content."},
                            {"role": "user", "content": f"Generate a search snippet (max 150 words) for this webpage that answers the query: '{query}'\n\nWebpage content: {content[:10000]}"}
                        ],
                        temperature=0.3,
                        max_tokens=250
                    )
                    
                    snippet = snippet_response.choices[0].message.content
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url
                    })
            
            return json.dumps({"results": results})
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return json.dumps({"results": []})
    
    @kernel_function(description="Get news articles about a company or topic")
    async def get_news(self, query: str, days: int = 30) -> str:
        """Get recent news about a company or topic using DuckDuckGo news search"""
        try:
            # Add time constraint to the query
            time_bound_query = f"{query} in the past {days} days"
            
            # Get URLs from DuckDuckGo news search
            urls = self.ddg_search.news_search(time_bound_query, 5)
            
            if not urls:
                # If DuckDuckGo search fails, use OpenAI to generate a response
                response = self.openai_client.chat.completions.create(
                    model=CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are a news search assistant that helps users find recent news articles."},
                        {"role": "user", "content": f"Find news articles from the past {days} days about: {query}. Since I couldn't access search results, provide your best knowledge about recent news on this topic."}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                generated_results = [
                    {
                        "title": f"Recent news about {query}",
                        "summary": response.choices[0].message.content[:300],
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "url": "No search results available"
                    }
                ]
                
                return json.dumps({"articles": generated_results})
            
            # Scrape content from the URLs
            articles = []
            for url in urls:
                scraped_result = self.scraper.scrape_website(url)
                
                if "content" in scraped_result and scraped_result["content"]:
                    content = scraped_result["content"]
                    
                    # Extract title from the URL if not in the content
                    title = extract_domain(url)
                    
                    # Get a summary using OpenAI
                    summary_response = self.openai_client.chat.completions.create(
                        model=CHAT_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": "You are a news summarizer. Create a concise, informative summary from news article content."},
                            {"role": "user", "content": f"Summarize this news article (max 150 words) about '{query}'\n\nArticle content: {content[:10000]}"}
                        ],
                        temperature=0.3,
                        max_tokens=250
                    )
                    
                    summary = summary_response.choices[0].message.content
                    
                    # Get the publication date (approximate)
                    date = datetime.now().strftime("%Y-%m-%d")  # Default to today
                    
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "date": date,
                        "url": url
                    })
            
            return json.dumps({"articles": articles})
        except Exception as e:
            logger.error(f"Error during news search: {str(e)}")
            return json.dumps({"articles": []})

class MarketTrendsPlugin:
    """Plugin for market trends analysis using DuckDuckGo search and OpenAI"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.ddg_search = ddg_search  
        self.scraper = WebContentScraper()
    
    @kernel_function(description="Analyze market trends and sentiment for an industry or company")
    async def analyze_market_trends(self, company: str, industry: str) -> str:
        """Analyze market trends for a company or industry using DuckDuckGo and OpenAI"""
        try:
            # Construct search queries for different aspects
            queries = [
                f"{company} {industry} market trends",
                f"{industry} industry trends latest",
                f"{company} recent news developments",
                f"{industry} market outlook",
                f"{company} competitive position {industry}"
            ]
            
            # Collect search results for all queries
            all_results = []
            for query in queries:
                urls = self.ddg_search.text_search(query, 2)
                for url in urls:
                    scraped_result = self.scraper.scrape_website(url)
                    if "content" in scraped_result and scraped_result["content"]:
                        text = scraped_result["content"]
                        all_results.append({
                            "query": query,
                            "url": url,
                            "content": text
                        })
            
            # If no results from DuckDuckGo, use OpenAI's knowledge
            if not all_results:
                response = self.openai_client.chat.completions.create(
                    model=CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are a market research analyst that identifies trends and sentiment."},
                        {"role": "user", "content": f"Analyze current market trends and sentiment for {company} in the {industry} industry without external search results. Use your knowledge to provide trends, sentiment score, and key points."}
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content
            
            # Combine all search results into a comprehensive analysis using OpenAI
            search_content = "\n\n".join([f"QUERY: {r['query']}\nURL: {r['url']}\nCONTENT: {r['content'][:1000]}" for r in all_results[:5]])
            
            response = self.openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a market research analyst that identifies trends and sentiment based on search results."},
                    {"role": "user", "content": f"""
                    Analyze the following search results about {company} in the {industry} industry.
                    
                    Provide a detailed JSON with:
                    1. A 'trends' array with relevant market trends
                    2. A 'sentiment_score' between -1.0 and 1.0
                    3. A 'key_points' array with important insights
                    
                    Base your analysis strictly on the search results provided.
                    
                    SEARCH RESULTS:
                    {search_content}
                    """
                    }
                ],
                temperature=0.5,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error during market analysis: {str(e)}")
            return json.dumps({"trends": [], "sentiment_score": 0, "key_points": []})

# ================ Agents ================

class CompetitorFinderAgent:
    """Agent to identify competitors for a given company URL"""
    
    def __init__(self, kernel, openai_client):
        self.kernel = kernel
        self.openai_client = openai_client
        self.ddg_search = ddg_search
        self.scraper = WebContentScraper()
        # Add plugins to the kernel
        self.web_scraper_plugin = kernel.add_plugin(WebScraperPlugin(openai_client), "WebScraper")
        self.web_search_plugin = kernel.add_plugin(WebSearchPlugin(openai_client), "WebSearch")
        
        # Create the agent
        self.agent = ChatCompletionAgent(
            name="CompetitorFinderAgent",
            # Use the explicit service creation with deployment_name instead of ai_model_id
            service=AzureChatCompletion(
                service_id="chat", 
                deployment_name=CHAT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            ),
            instructions="""
            You are an expert competitor intelligence analyst with 10+ years of experience identifying market rivals.

            OBJECTIVE: Identify the most relevant competitors for a given company based on their website.

            PROCESS:
            1. FIRST analyze the source company's website to identify:
            - Core products/services offered
            - Target customer segments 
            - Value proposition and positioning
            - Geographic markets served
            - Company size/scale indicators

            2. THEN analyze the provided DuckDuckGo search results to find competitors using these criteria:
            - Direct competitors (same products/services, same market segments)
            - Indirect competitors (similar solutions to the same customer needs)
            - Emerging disruptors in the same space
            - Market leaders and notable innovators

            3. For EACH competitor, gather and provide:
            - Full company name with proper capitalization
            - Official website URL (validate the URL is correct)
            - Similarity score (0.0-1.0) based on: market overlap (40%), product similarity (40%), size comparability (20%)
            - Concise description (30-60 words) highlighting their main differentiators and competitive position

            4. IMPORTANT: Only include competitors you are confident are real companies. DO NOT generate placeholder competitors or example companies. If you cannot find enough legitimate competitors, return fewer entries rather than creating fictional ones.

            RESPONSE FORMAT: Return ONLY a JSON object with the 'competitors' array. Each competitor must include all four fields (name, url, similarity_score, description). Sort by similarity_score in descending order.

            {
                "competitors": [
                    {
                        "name": "Competitor Name",
                        "url": "https://competitor-website.com",
                        "similarity_score": 0.9,
                        "description": "Detailed description focusing on how they compete with the source company and their key differentiators."
                    }
                ]
            }
            """,
            plugins=[self.web_scraper_plugin, self.web_search_plugin],
        )
    
    async def find_competitors(self, company_url: str, count: int = 3) -> List[CompetitorInfo]:
        """Find top competitors for the given company URL"""
        try:
            # First, get information about the company itself
            content = await fetch_url(company_url)
            company_info = extract_company_info(content)
            company_name = company_info.get("name", extract_domain(company_url))
            company_description = company_info.get("description", "")
            
            # Collect search results from DuckDuckGo for different aspects
            search_queries = [
                f"{company_name} competitors",
                f"companies similar to {company_name}",
            ]
            
            # Collect search results from all queries
            all_search_results = []
            for query in search_queries:
                urls = self.ddg_search.text_search(query, 3)
                for url in urls:
                    all_search_results.append({"query": query, "url": url})
            
            # Scrape content from the URLs for context
            search_content = ""
            for i, item in enumerate(all_search_results[:10], 1):
                scraped_result = self.scraper.scrape_website(item["url"])
                if "content" in scraped_result and scraped_result["content"]:
                    # Get a short excerpt
                    excerpt = scraped_result["content"][:2000]
                    search_content += f"SOURCE {i} (QUERY: {item['query']}, URL: {item['url']}):\n{excerpt}\n\n"
            
            # Create query for the agent including the search results
            task = f"""Find the top {count} competitors for {company_name} (URL: {company_url}).
            
            Company information:
            {company_description}
            
            DuckDuckGo search results:
            {search_content}
            
            Remember to ONLY include real, identifiable competitors. DO NOT create fictional or placeholder competitors. If you can't find enough legitimate competitors, return fewer than {count}."""
            
            # Create a new thread for the conversation
            history = ChatHistory()
            history.add_user_message(task)
            
            # Invoke the agent
            runtime = InProcessRuntime()
            runtime.start()
            
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = await self.agent.get_response(messages=task)
                    break
                except Exception as e:
                    logger.warning(f"Agent response error (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)  # Wait before retrying
            
            if not response:
                logger.error("Failed to get agent response after retries")
                return []
            
            await runtime.stop_when_idle()
            
            # Parse the response using the safer method
            response_text = response.message.content
            data = await safe_parse_json_from_agent(response_text)
            
            # Extract competitors
            competitors = []
            if isinstance(data, dict) and "competitors" in data:
                competitors_data = data["competitors"]
            elif isinstance(data, list):
                competitors_data = data
            else:
                competitors_data = []
                
            for comp in competitors_data:
                if isinstance(comp, dict) and "name" in comp and "url" in comp:
                    competitors.append(CompetitorInfo(
                        name=comp.get("name", "Unknown"),
                        url=comp.get("url", ""),
                        similarity_score=float(comp.get("similarity_score", 0.5)),
                        description=comp.get("description", "")
                    ))
            
            return competitors  # Note: No slicing to limit results, will return however many real competitors were found
        
        except Exception as e:
            logger.error(f"Error in CompetitorFinderAgent: {str(e)}")
            return []
class ProductScraperAgent:
    """Agent to extract product and pricing information from competitor websites"""
    
    def __init__(self, kernel, openai_client):
        self.kernel = kernel
        self.openai_client = openai_client
        self.ddg_search = ddg_search
        self.scraper = WebContentScraper()
        self.web_scraper_plugin = kernel.add_plugin(WebScraperPlugin(openai_client), "WebScraper")
        # Add plugins to the kernel
        
        # Create the agent with improved prompt
        self.agent = ChatCompletionAgent(
            name="ProductScraperAgent",
            # Use the explicit service creation with deployment_name instead of ai_model_id
            service=AzureChatCompletion(
                service_id="chat", 
                deployment_name=CHAT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            ),
            instructions="""
            You are an expert product data extraction specialist with 10+ years of e-commerce analysis experience.

            OBJECTIVE: Extract comprehensive, structured product information from company websites and search results.

            PROCESS:
            1. FIRST identify product pages or sections from the provided website content and search results:
               - Look for navigation links to product catalogs, solutions, or offerings
               - Check for "Products", "Shop", "Solutions", "Services" sections
               - Identify pricing pages or feature comparison tables
               - Carefully analyze the DuckDuckGo search results for product information

            2. For EACH product identified, extract these REQUIRED fields:
               - name: Full product name exactly as displayed (string)
               - price: Numeric price value WITHOUT currency symbols (number, e.g. 99.99, not "$99.99")
               - url: Complete URL to product page (string with full URL)
               - id: Generate a unique identifier for each product (string)
               - last_updated: Current date in YYYY-MM-DD format (string)
               
            3. Also extract these OPTIONAL fields:
               - description: Product description (string)
               - features: Product features as an array of strings (array of strings)

            4. Apply these VALIDATION STANDARDS:
               - ALWAYS include all REQUIRED fields for each product
               - For B2B products without visible pricing, use 0.0 for price
               - If a product has no features, use an empty array []
               - ALWAYS format price as a numeric value (e.g., 99.99), NEVER as a string
               - ALWAYS format features as an array of strings, even if only one feature

            5. IMPORTANT: Only extract products you are confident actually exist. DO NOT generate placeholder or example products. If you cannot identify any real products, return an empty products array.

            If the website is blocked or unavailable, rely on the search results provided.
            If no products can be identified from either the website or search results, return an empty array.

            RESPONSE FORMAT: Return ONLY a JSON object with the 'products' array:

            {
                "products": [
                    {
                        "id": "unique-id",
                        "name": "Product Name",
                        "price": 99.99,
                        "description": "Product description",
                        "url": "https://product-url.com",
                        "features": ["Feature 1", "Feature 2"],
                        "last_updated": "2025-05-19"
                    }
                ]
            }
            """,
            plugins=[self.web_scraper_plugin],
        )
    
    async def extract_products(self, company_url: str) -> List[ProductInfo]:
        """Extract products from a company website with improved validation error handling"""
        try:
            # Check if domain is blocked before even trying
            domain = extract_domain(company_url)
            if domain in BLOCKED_DOMAINS:
                logger.warning(f"Skipping product extraction for {company_url} as domain is blocked")
                return []
                
            # First, check if we can access the site at all
            content = await fetch_url(company_url)
            homepage_content = content if content else ""
            
            # Perform DuckDuckGo searches to find product-related pages
            search_queries = [
                f"site:{domain} products",
                f"site:{domain} pricing",
                f"site:{domain} solutions"
            ]
            
            # Collect search results from all queries
            all_search_results = []
            for query in search_queries:
                urls = self.ddg_search.text_search(query, 2)  # Get 2 results per query
                for url in urls:
                    if url not in [r["url"] for r in all_search_results]:
                        all_search_results.append({"query": query, "url": url})
            
            # Scrape content from the search results
            search_content = ""
            for i, item in enumerate(all_search_results[:6], 1):  # Limit to top 6 results to avoid token limits
                scraped_result = self.scraper.scrape_website(item["url"])
                if "content" in scraped_result and scraped_result["content"]:
                    # Get a short excerpt
                    excerpt = scraped_result["content"][:2000]  # Limit each excerpt to 2000 characters
                    search_content += f"SOURCE {i} (QUERY: {item['query']}, URL: {item['url']}):\n{excerpt}\n\n"
            
            # If we couldn't access the homepage or find any search results
            if not homepage_content and not search_content:
                logger.warning(f"Unable to access {company_url} or find any product information through search")
                return []
                
            # Proceed with product extraction using both homepage content and search results
            task = f"""Extract product information from {company_url}.

Homepage content:
{homepage_content[:5000]}  

DuckDuckGo search results for product pages:
{search_content}

Remember to ONLY extract real, identifiable products. DO NOT create fictional or placeholder products. If you can't find any legitimate products, return an empty products array."""
            
            # Create a new thread for the conversation
            history = ChatHistory()
            history.add_user_message(task)
            
            # Invoke the agent with retries
            runtime = InProcessRuntime()
            runtime.start()
            
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = await self.agent.get_response(messages=task)
                    break
                except Exception as e:
                    logger.warning(f"Agent response error (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)  # Wait before retrying
            
            if not response:
                logger.error("Failed to get agent response after retries")
                return []
            
            await runtime.stop_when_idle()
            
            # Parse the response using the safer method
            response_text = response.message.content
            data = await safe_parse_json_from_agent(response_text)
            
            # Parse products
            products = []
            if isinstance(data, dict) and "products" in data:
                products_data = data["products"]
            elif isinstance(data, list):
                products_data = data
            else:
                products_data = []
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            for prod in products_data:
                if isinstance(prod, dict):
                    try:
                        # Ensure all required fields are present with proper defaults
                        product_dict = {
                            # Required fields with defaults
                            "id": prod.get("id", str(uuid.uuid4())[:8]),
                            "name": prod.get("name", "Unknown Product"),
                            "price": 0.0,  # Default price, will be updated if present
                            "url": prod.get("url", company_url),
                            "last_updated": prod.get("last_updated", current_date),
                            
                            # Optional fields with defaults
                            "description": prod.get("description", ""),
                            "features": []  # Default empty list
                        }
                        
                        # Handle price conversion safely
                        if "price" in prod:
                            try:
                                price_str = str(prod["price"]).replace('$', '').replace('€', '').replace('£', '').strip()
                                product_dict["price"] = float(price_str) if price_str else 0.0
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid price format for product {product_dict['name']}, using 0.0")
                                product_dict["price"] = 0.0
                        
                        # Handle features safely
                        if "features" in prod:
                            if isinstance(prod["features"], list):
                                product_dict["features"] = prod["features"]
                            elif isinstance(prod["features"], str):
                                # If features was provided as a string, convert to a single-item list
                                product_dict["features"] = [prod["features"]]
                        
                        # Create ProductInfo object with the validated dictionary
                        products.append(ProductInfo(**product_dict))
                    except Exception as e:
                        logger.error(f"Validation error for product {prod.get('name', 'Unknown')}: {str(e)}")
                        # Continue with other products rather than failing completely
            
            logger.info(f"Successfully extracted {len(products)} products from {company_url}")
            return products
        
        except Exception as e:
            logger.error(f"Error in ProductScraperAgent for {company_url}: {str(e)}")
            return []
class MarketSentimentAgent:
    """Agent to analyze market sentiment and trends"""
    
    def __init__(self, kernel, openai_client):
        self.kernel = kernel
        self.openai_client = openai_client
        self.ddg_search = ddg_search  
        self.scraper = WebContentScraper()
        
        # Add plugins to the kernel
        self.web_search_plugin = kernel.add_plugin(WebSearchPlugin(openai_client), "WebSearch")
        self.market_trends_plugin = kernel.add_plugin(MarketTrendsPlugin(openai_client), "MarketTrends")
        self.web_scraper_plugin = kernel.add_plugin(WebScraperPlugin(openai_client), "WebScraper")
        
        # Create the agent
        self.agent = ChatCompletionAgent(
            name="MarketSentimentAgent",
            # Use the explicit service creation with deployment_name instead of ai_model_id
            service=AzureChatCompletion(
                service_id="chat", 
                deployment_name=CHAT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            ),
            instructions="""
                You are a market intelligence analyst with expertise in sentiment analysis and trend identification.

                OBJECTIVE: Provide quantitative and qualitative analysis of market sentiment and emerging trends for a company/industry.

                DATA COLLECTION PROCESS:
                1. Analyze the provided search results and sources for:
                - Financial news and analyst reports
                - Industry publications and trade journals
                - Social media conversations and forums
                - Product reviews and customer feedback
                - Press releases and corporate announcements

                2. For SENTIMENT ANALYSIS:
                - Calculate overall sentiment on scale from -1.0 (extremely negative) to 1.0 (extremely positive)
                - Identify primary drivers of positive sentiment
                - Identify primary drivers of negative sentiment
                - Note any significant sentiment shifts in the past 30/60/90 days
                - Document specific evidence supporting your assessment

                3. For TREND IDENTIFICATION:
                - Prioritize trends by relevance score (0.0-1.0) to the specific company/industry
                - Consider technological, regulatory, competitive, and consumer behavior trends
                - Assess trend maturity (emerging, growing, peaking, declining)
                - Evaluate trend impact on business outcomes
                - Document specific evidence and sources for each trend

                RESPONSE FORMAT: Return ONLY a JSON object with BOTH 'market_sentiment' and 'market_trends' sections:

                {
                    "market_sentiment": {
                        "overall_sentiment": 0.7,
                        "key_points": ["Specific driver of sentiment 1", "Specific driver of sentiment 2"],
                        "sources": ["Specific source 1", "Specific source 2"]
                    },
                    "market_trends": [
                        {
                            "trend": "Specific Trend Name",
                            "sources": ["Specific source 1", "Specific source 2"],
                            "relevance_score": 0.9,
                            "description": "Detailed description of the trend and its implications"
                        }
                    ]
                }

                CRITICAL: Always use numerical values for sentiment and relevance scores. Provide at least 3-5 substantive trends sorted by relevance_score.
                """,
            plugins=[self.web_search_plugin, self.market_trends_plugin],
        )
    
    async def analyze_sentiment(self, company_name: str, industry: str) -> tuple[MarketSentiment, List[MarketTrend]]:
        """Analyze market sentiment and trends for a company and industry"""
        try:
            # Collect search results from DuckDuckGo for different aspects
            search_queries = [
                f"{company_name} {industry} market sentiment",
                f"{company_name} recent news",
                f"{industry} trends",
            ]
            
            # Collect search results for all queries
            all_search_results = []
            for query in search_queries:
                urls = self.ddg_search.text_search(query, 2)
                for url in urls:
                    all_search_results.append({"query": query, "url": url})
            
            # Remove duplicates while preserving order
            unique_urls = []
            for item in all_search_results:
                if item["url"] not in [u["url"] for u in unique_urls]:
                    unique_urls.append(item)
            
            # Fetch content from the URLs
            search_content = ""
            for i, item in enumerate(unique_urls[:10], 1):
                scraped_result = self.scraper.scrape_website(item["url"])
                if "content" in scraped_result and scraped_result["content"]:
                    # Get a short excerpt
                    excerpt = scraped_result["content"][:2500]
                    search_content += f"SOURCE {i} (QUERY: {item['query']}, URL: {item['url']}):\n{excerpt}\n\n"
            
            # Create a task that includes the search results
            task = f"""
            Analyze market sentiment and trends for {company_name} in the {industry} industry based on these search results:
            
            {search_content}
            
            Return a JSON object with:
            1. market_sentiment: An object containing:
               - overall_sentiment: A NUMERICAL value between -1.0 (extremely negative) and 1.0 (extremely positive)
               - key_points: Array of key points about market sentiment
               - sources: Array of sources used for sentiment analysis
            
            2. market_trends: Array of trend objects, each containing:
               - trend: The name of the trend
               - sources: Array of sources supporting this trend
               - relevance_score: A NUMERICAL value between 0.0 and 1.0 indicating relevance
               - description: Brief description of the trend
            
            You MUST format numbers as floats, not strings. For example, use 0.7 not "0.7" or "positive".
            If you cannot determine sentiment or trends from the sources, use neutral values and indicate limitations in your response.
            """
            
            # Create a new thread for the conversation
            history = ChatHistory()
            history.add_user_message(task)
            
            # Invoke the agent with retries
            runtime = InProcessRuntime()
            runtime.start()
            
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = await self.agent.get_response(messages=task)
                    break
                except Exception as e:
                    logger.warning(f"Agent response error (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)  # Wait before retrying
            
            if not response:
                logger.error("Failed to get agent response after retries")
                default_sentiment = MarketSentiment(
                    overall_sentiment=0.0,
                    key_points=["Failed to analyze sentiment"],
                    sources=[]
                )
                return default_sentiment, []
            
            await runtime.stop_when_idle()
            
            # Parse the response using the safer method
            response_text = response.message.content
            data = await safe_parse_json_from_agent(response_text)
            
            # Extract market sentiment with proper type conversion
            sentiment_data = data.get("market_sentiment", {})
            
            # Ensure sentiment score is a float
            sentiment_score = sentiment_data.get("overall_sentiment", 0.0)
            if isinstance(sentiment_score, str):
                # Convert textual sentiment to numeric value if needed
                if sentiment_score.lower() == 'positive':
                    sentiment_score = 0.7
                elif sentiment_score.lower() == 'very positive':
                    sentiment_score = 0.9
                elif sentiment_score.lower() == 'neutral':
                    sentiment_score = 0.0
                elif sentiment_score.lower() == 'negative':
                    sentiment_score = -0.7
                elif sentiment_score.lower() == 'very negative':
                    sentiment_score = -0.9
                else:
                    # Try to convert string to float, fallback to 0.0
                    try:
                        sentiment_score = float(sentiment_score)
                    except ValueError:
                        logger.warning(f"Could not convert sentiment score '{sentiment_score}' to float. Using default 0.0")
                        sentiment_score = 0.0
            
            # Ensure sentiment is within -1.0 to 1.0 range
            sentiment_score = max(-1.0, min(1.0, float(sentiment_score)))
            
            sentiment = MarketSentiment(
                overall_sentiment=sentiment_score,
                key_points=sentiment_data.get("key_points", []),
                sources=sentiment_data.get("sources", [])
            )
            
            # Extract market trends with proper type conversion
            trends_data = data.get("market_trends", [])
            trends = []
            for trend_data in trends_data:
                # Ensure relevance score is a float
                relevance_score = trend_data.get("relevance_score", 0.5)
                if isinstance(relevance_score, str):
                    try:
                        relevance_score = float(relevance_score)
                    except ValueError:
                        logger.warning(f"Could not convert relevance score '{relevance_score}' to float. Using default 0.5")
                        relevance_score = 0.5
                
                # Ensure relevance is within 0.0 to 1.0 range
                relevance_score = max(0.0, min(1.0, float(relevance_score)))
                
                trends.append(MarketTrend(
                    trend=trend_data.get("trend", ""),
                    sources=trend_data.get("sources", []),
                    relevance_score=relevance_score,
                    description=trend_data.get("description", "")
                ))
            
            return sentiment, trends
        
        except Exception as e:
            logger.error(f"Error in MarketSentimentAgent: {str(e)}")
            default_sentiment = MarketSentiment(
                overall_sentiment=0.0,
                key_points=["Error analyzing sentiment"],
                sources=[]
            )
            return default_sentiment, []

class InsightGeneratorAgent:
    """Agent to generate insights and recommendations"""
    
    def __init__(self, kernel, openai_client):
        self.kernel = kernel
        self.openai_client = openai_client
        self.ddg_search = ddg_search  
        self.scraper = WebContentScraper()
        self.web_scraper_plugin = kernel.add_plugin(WebScraperPlugin(openai_client), "WebScraper")
        self.web_search_plugin = kernel.add_plugin(WebSearchPlugin(openai_client), "WebSearch")
        self.market_trends_plugin = kernel.add_plugin(MarketTrendsPlugin(openai_client), "MarketTrends")
        # Create the agent
        self.agent = ChatCompletionAgent(
            name="InsightGeneratorAgent",
            # Use the explicit service creation with deployment_name instead of ai_model_id
            service=AzureChatCompletion(
                service_id="chat", 
                deployment_name=CHAT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            ),
            instructions="""
                You are a strategic business consultant with expertise in competitive analysis and market strategy.

                OBJECTIVE: Transform raw market and competitor data into actionable business insights and strategic recommendations.

                ANALYTICAL PROCESS:
                1. FIRST compare the target company with competitors across:
                - Product offerings, features, and pricing
                - Market positioning and value propositions
                - Strengths, weaknesses, and unique advantages
                - Customer feedback and sentiment patterns

                2. THEN analyze market dynamics using both the provided data and search results:
                - Identify gaps between customer needs and available solutions
                - Spot pricing inefficiencies or opportunity zones
                - Recognize emerging trends that create new opportunities
                - Detect potential threats from market shifts or competitor actions

                3. FOR EACH INSIGHT:
                - Ensure it is specific, data-backed, and non-obvious
                - Quantify business impact where possible (market share, revenue)
                - Assign relevance score based on potential impact and urgency
                - Develop 2-3 concrete action items that directly address the insight
                - Link insights to specific competitive advantages or disadvantages

                4. FOR RECOMMENDATIONS:
                - Provide specific, actionable guidance (not general advice)
                - Focus on practical, implementable actions within 30/90/180 days
                - Consider resource requirements and implementation difficulty
                - Prioritize recommendations with highest ROI potential
                - Address both defensive (threat mitigation) and offensive (opportunity capture) strategies

                RESPONSE FORMAT: Return ONLY a JSON object with both 'insights' and 'recommendations' sections:

                {
                    "insights": [
                        {
                            "insight": "Specific, non-obvious insight with quantified impact potential",
                            "relevance": 0.9,
                            "action_items": ["Specific action 1 with expected outcome", "Specific action 2 with expected outcome"]
                        }
                    ],
                    "recommendations": [
                        "Specific, actionable recommendation with timeframe and expected outcome",
                        "Next specific, actionable recommendation with resources needed and priority level"
                    ]
                }
                CRITICAL: Provide at least 3-5 substantial insights and 5-7 prioritized recommendations. Ensure all content is specific to the target company's situation, not generic advice.
                """,
        )
    
    async def generate_insights(self, 
                                company_name: str,
                                competitors: List[CompetitorInfo],
                                products: Dict[str, List[ProductInfo]],
                                market_sentiment: MarketSentiment,
                                market_trends: List[MarketTrend]) -> tuple[List[InsightItem], List[str]]:
        """Generate insights and recommendations based on the gathered data"""
        try:
            # Prepare data for the agent
            competitors_json = [comp.model_dump() for comp in competitors]
            
            products_json = {}
            for company, prods in products.items():
                products_json[company] = [prod.model_dump() for prod in prods]
            
            sentiment_json = market_sentiment.model_dump()
            trends_json = [trend.model_dump() for trend in market_trends]
            
            # Collect search results from DuckDuckGo for industry insights
            search_queries = [
                f"{company_name} business strategy",
                f"{company_name} competitive advantage",
                f"innovation in {' '.join([t.trend for t in market_trends[:2]])}"
            ]
            
            # Collect search results for all queries
            all_search_results = []
            for query in search_queries:
                urls = self.ddg_search.text_search(query, 2)
                for url in urls:
                    all_search_results.append({"query": query, "url": url})
            
            # Remove duplicates while preserving order
            unique_urls = []
            for item in all_search_results:
                if item["url"] not in [u["url"] for u in unique_urls]:
                    unique_urls.append(item)
            
            # Fetch content from the URLs
            search_content = ""
            for i, item in enumerate(unique_urls[:8], 1):
                scraped_result = self.scraper.scrape_website(item["url"])
                if "content" in scraped_result and scraped_result["content"]:
                    # Get a short excerpt
                    excerpt = scraped_result["content"][:2000]
                    search_content += f"SOURCE {i} (QUERY: {item['query']}, URL: {item['url']}):\n{excerpt}\n\n"
            
            # Create the agent task with structured data and search results
            data_json = json.dumps({
                "company_name": company_name,
                "competitors": competitors_json,
                "products": products_json,
                "market_sentiment": sentiment_json,
                "market_trends": trends_json
            }, indent=2)
            
            task = f"""
            Generate strategic insights and recommendations based on the following competitor and market analysis:
            
            ANALYSIS DATA:
            {data_json}
            
            ADDITIONAL MARKET RESEARCH:
            {search_content}
            
            Return a JSON object with:
            1. "insights": An array of insight objects, each with "insight", "relevance" (0-1), and "action_items" fields
            2. "recommendations": An array of specific, actionable recommendations
            
            Focus on practical, actionable insights that can drive business decisions. Use both the structured analysis data and the market research sources to provide comprehensive insights.
            """
            
            # Invoke the agent with retries
            runtime = InProcessRuntime()
            runtime.start()
            
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = await self.agent.get_response(messages=task)
                    break
                except Exception as e:
                    logger.warning(f"Agent response error (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)  # Wait before retrying
            
            if not response:
                logger.error("Failed to get agent response after retries")
                return [], ["No recommendations could be generated due to an error"]
            
            await runtime.stop_when_idle()
            
            # Parse the response using the safer method
            response_text = response.message.content
            data = await safe_parse_json_from_agent(response_text)
            
            # Extract insights
            insights_data = data.get("insights", [])
            insights = []
            for insight_data in insights_data:
                # Ensure relevance is a float
                relevance = insight_data.get("relevance", 0.5)
                if isinstance(relevance, str):
                    try:
                        relevance = float(relevance)
                    except ValueError:
                        relevance = 0.5
                
                # Ensure relevance is within 0.0 to 1.0 range
                relevance = max(0.0, min(1.0, float(relevance)))
                
                insights.append(InsightItem(
                    insight=insight_data.get("insight", ""),
                    relevance=relevance,
                    action_items=insight_data.get("action_items", [])
                ))
            
            # Extract recommendations
            recommendations = data.get("recommendations", [])
            
            return insights, recommendations
        
        except Exception as e:
            logger.error(f"Error in InsightGeneratorAgent: {str(e)}")
            return [], ["No recommendations could be generated due to an error"]

# ================ Orchestration ================

async def run_analysis(task_id: str, url: str, depth: int):
    """Run the full competitor and market analysis workflow with improved error handling and parallelization"""
    try:
        # Update task status
        tasks[task_id] = {
            "status": "in_progress",
            "progress": 0.1,
            "message": "Starting analysis",
            "result": None
        }
        
        # Initialize the main kernel
        kernel = Kernel()
        
        # Add Azure OpenAI services
        service_id = "chat"
        embedding_service_id = "embeddings"
        
        kernel.add_service(
            AzureChatCompletion(
                service_id=service_id,
                deployment_name=CHAT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            )
        )
        
        
        embedding_gen = AzureTextEmbedding(
                service_id=embedding_service_id,
                deployment_name=EMBEDDING_DEPLOYMENT,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            )
        kernel.add_service(embedding_gen)
            
        # Initialize memory
        memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_gen)
        kernel.add_plugin(TextMemoryPlugin(memory), "memory")
            
        # Initialize OpenAI client for direct API calls
        openai_client = AzureOpenAI(
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY
            )
            
        # Initialize DuckDuckGo search manager
        ddg_search = DuckDuckGoSearchManager()
            
        # Initialize web scraper
        scraper = WebContentScraper()
            
        # Update task status
        tasks[task_id]["progress"] = 0.2
        tasks[task_id]["message"] = "Initializing agents"
            
        # Initialize agents
        competitor_agent = CompetitorFinderAgent(kernel, openai_client)
        product_agent = ProductScraperAgent(kernel, openai_client)
        sentiment_agent = MarketSentimentAgent(kernel, openai_client)
        insight_agent = InsightGeneratorAgent(kernel, openai_client)
            
        # Step 1: Get information about the target company
        tasks[task_id]["progress"] = 0.3
        tasks[task_id]["message"] = "Analyzing target company"
            
        content = None
        scraped_result = scraper.scrape_website(url)
        if "content" in scraped_result and scraped_result["content"]:
            content = scraped_result["content"]
        else:
            # If scraper failed, try with async fetch_url
            content = await fetch_url(url)
                
        company_info = extract_company_info(content) if content else {"name": extract_domain(url), "description": ""}
        company_name = company_info.get("name", extract_domain(url))
            
        # If we couldn't get company info from the website, use DuckDuckGo search as fallback
        if company_name == "Unknown" or not company_info.get("description"):
            domain = extract_domain(url)
            search_results = ddg_search.text_search(f"about {domain} company", 3)
                
            if search_results:
                # Use OpenAI to generate a description based on the search results
                search_content = f"Search results about {domain}:\n\n"
                for i, search_url in enumerate(search_results, 1):
                    search_content += f"{i}. {search_url}\n"
                        
                    # Try to scrape the content of each search result
                    try:
                        search_scraped = scraper.scrape_website(search_url)
                        if "content" in search_scraped and search_scraped["content"]:
                            search_content += f"\nContent: {search_scraped['content'][:1000]}\n\n"
                    except Exception as e:
                        logger.error(f"Error scraping search result {search_url}: {str(e)}")
                    
                response = openai_client.chat.completions.create(
                        model=CHAT_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": "You are an expert business analyst. Extract company information from search results."},
                            {"role": "user", "content": f"Based on these search results about {domain}, provide the company name and a detailed description of what they do:\n\n{search_content}"}
                        ],
                        temperature=0.5,
                        max_tokens=500
                    )
                    
                generated_info = response.choices[0].message.content
                    
                # Parse the generated info to extract company name and description
                try:
                    if "Company Name:" in generated_info:
                        name_desc_parts = generated_info.split("Company Name:", 1)[1].strip()
                        if "Description:" in name_desc_parts:
                            name_parts = name_desc_parts.split("Description:", 1)
                            company_name = name_parts[0].strip()
                            company_info["description"] = name_parts[1].strip()
                        else:
                            company_name = name_desc_parts
                    elif "\n" in generated_info:
                        lines = generated_info.strip().split("\n")
                        company_name = lines[0].strip()
                        company_info["description"] = "\n".join(lines[1:]).strip()
                except:
                    # If parsing fails, use the whole text as description
                    company_info["description"] = generated_info
            
        company_domain = extract_domain(url)
            
        # Step 2: Run target product extraction and competitor finding IN PARALLEL
        tasks[task_id]["progress"] = 0.4
        tasks[task_id]["message"] = f"Finding competitors and products for {company_name}"
            
        # Run target product extraction and competitor finding in parallel
        target_products_task = product_agent.extract_products(url)
        competitors_task = competitor_agent.find_competitors(url, count=5)
            
        target_products, competitors = await asyncio.gather(
                target_products_task,
                competitors_task
            )
            
        # Step 3: Extract products from competitors IN PARALLEL
        tasks[task_id]["progress"] = 0.5
        tasks[task_id]["message"] = "Analyzing competitor products"
            
        all_products = {company_name: target_products}
            
        # Define an async function to extract products safely
        async def extract_competitor_products(competitor):
            try:
                domain = extract_domain(competitor.url)
                if domain in BLOCKED_DOMAINS:
                    logger.warning(f"Skipping competitor {competitor.name} ({competitor.url}) as domain is blocked")
                    return competitor.name, []
                        
                # Add a domain-specific check to avoid repeated failures
                domain_failure_count = sum(1 for url in FAILED_URLS if domain in url)
                if domain_failure_count > 5:  # If we've had multiple failures for this domain
                    logger.warning(f"Too many failed URLs for {domain}, skipping further extraction")
                    return competitor.name, []
                        
                competitor_products = await product_agent.extract_products(competitor.url)
                return competitor.name, competitor_products
            except Exception as e:
                logger.error(f"Error extracting products for {competitor.name}: {str(e)}")
                return competitor.name, []
            
        # Run all competitor product extractions in parallel
        # competitor_product_tasks = [extract_competitor_products(comp) for comp in competitors]
        # competitor_products_results = await asyncio.gather(*competitor_product_tasks)
        # Skip competitor product extraction, just create empty product lists for competitors
        competitor_products_results = [(comp.name, []) for comp in competitors]
        # Update all_products dictionary with results
        for comp_name, comp_products in competitor_products_results:
            all_products[comp_name] = comp_products
            
        # Step 4: Analyze market sentiment and trends
        tasks[task_id]["progress"] = 0.7
        tasks[task_id]["message"] = "Analyzing market sentiment and trends"
            
        # Determine industry based on company and competitors
        industry_prompt = f"""
        Based on the company {company_name} and its competitors {', '.join([c.name for c in competitors])},
        determine the primary industry category. Return just the industry name, nothing else.
            """
            
        industry_response = openai_client.chat.completions.create(
               model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert at categorizing businesses into industries."},
                    {"role": "user", "content": industry_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
        industry = industry_response.choices[0].message.content.strip()
            
        market_sentiment, market_trends = await sentiment_agent.analyze_sentiment(company_name, industry)
            
        # Step 5: Generate insights and recommendations
        tasks[task_id]["progress"] = 0.9
        tasks[task_id]["message"] = "Generating insights and recommendations"
            
        insights, recommendations = await insight_agent.generate_insights(
                company_name, competitors, all_products, market_sentiment, market_trends
            )
            
        report = CompetitorAnalysisReport(
                timestamp=datetime.now().isoformat(),
                company_url=url,
                company_name=company_name,
                competitors=competitors,
                products=all_products,
                market_trends=market_trends,
                market_sentiment=market_sentiment,
                insights=insights,
                recommendations=recommendations
            )

        # Save the report using the save_task_result function (no need to save it twice)
        report_filename = save_task_result(task_id, report)

        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 1.0
        tasks[task_id]["message"] = "Analysis completed successfully"
        tasks[task_id]["result"] = report
        tasks[task_id]["completion_time"] = time.time()
        tasks[task_id]["report_filename"] = report_filename  # Store the filename for reference
        return report
        
    except Exception as e:
        logger.error(f"Error in analysis workflow: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Analysis failed: {str(e)}"
        tasks[task_id]["completion_time"] = time.time()
        return None

# ================ API Endpoints ================

@app.post("/analyze", response_model=CompetitorResponse)
async def analyze_competitor(request: CompetitorRequest, background_tasks: BackgroundTasks):
    """Start the competitor analysis process"""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Task queued",
        "result": None
    }
    
    background_tasks.add_task(run_analysis, task_id, str(request.url), request.depth)
    
    return CompetitorResponse(task_id=task_id)
# Add this constant at the top with other constants
TASK_RETENTION_HOURS = 2  # How long to keep completed tasks in memory

# Add this function for cleanup
def cleanup_old_tasks():
    """Remove completed tasks from memory after retention period"""
    current_time = time.time()
    to_remove = []
    
    for task_id, task_info in tasks.items():
        # Only clean up completed or failed tasks
        if task_info["status"] in ["completed", "failed"]:
            # Check if the task has completion_time and if retention period has passed
            if "completion_time" in task_info and (current_time - task_info["completion_time"]) > (TASK_RETENTION_HOURS * 3600):
                to_remove.append(task_id)
    
    # Remove the old tasks
    for task_id in to_remove:
        del tasks[task_id]
        logger.info(f"Cleaned up completed task {task_id} from memory during scheduled cleanup")
@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a task"""
    # First check if the task is in memory
    if task_id in tasks:
        task_info = tasks[task_id]
        
        # If task is completed or failed, check if it's time to clean it up
        if task_info["status"] in ["completed", "failed"] and "completion_time" in task_info:
            current_time = time.time()
            time_elapsed = current_time - task_info["completion_time"]
            
            # If retention period has passed (default 2 hours), clean up from memory
            if time_elapsed > (TASK_RETENTION_HOURS * 3600):
                # Save the result for return before removing from memory
                result = task_info.get("result")
                # Remove from memory
                del tasks[task_id]
                logger.info(f"Cleaned up completed task {task_id} from memory during status check")
                
                # If we have a result, return it
                if result:
                    return TaskStatus(
                        task_id=task_id,
                        status="completed",
                        progress=1.0,
                        message="Analysis completed successfully (restored from persistent storage)",
                        result=result
                    )
        
        # Return the task info from memory
        return TaskStatus(
            task_id=task_id,
            status=task_info["status"],
            progress=task_info["progress"],
            message=task_info["message"],
            result=task_info["result"]
        )
    
    # If not in memory, check if completed result exists on disk
    result = load_task_result(task_id)
    if result:
        return TaskStatus(
            task_id=task_id,
            status="completed",
            progress=1.0,
            message="Analysis completed successfully (restored from persistent storage)",
            result=result
        )
    
    # Run general cleanup of old tasks while we're here
    cleanup_old_tasks()
    
    # If neither in memory nor on disk, the task doesn't exist
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/reports", response_model=List[str])
async def list_reports():
    """Get a list of all available reports"""
    reports = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            reports.append(filename)
    return reports

@app.get("/reports/{report_name}")
async def get_report(report_name: str):
    """Get a specific report by name"""
    filepath = os.path.join(DATA_DIR, report_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(filepath, 'r') as f:
        report_data = json.load(f)
    
    return JSONResponse(content=report_data)

@app.get("/", include_in_schema=False)
async def read_root():
    """Root endpoint that returns app info"""
    return {
        "app": "Competitor Intelligence & Market Trends Agent",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/analyze", "method": "POST", "description": "Start a new competitor analysis"},
            {"path": "/status/{task_id}", "method": "GET", "description": "Check the status of an analysis task"},
            {"path": "/reports", "method": "GET", "description": "List all available reports"},
            {"path": "/reports/{report_name}", "method": "GET", "description": "Get a specific report"}
        ]
    }

# Entry point for the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
