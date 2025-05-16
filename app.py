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

# Environment variables - in production, set these as app settings in Azure App Service
AZURE_OPENAI_ENDPOINT = "https://prodhubfinnew-openai-97de.openai.azure.com/"
AZURE_OPENAI_API_KEY = "97fa8c02f9e64e8ea5434987b11fe6f4"  # In production, use environment variables
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
CHAT_DEPLOYMENT = "gpt-4.1"  # Replace with your actual deployment name

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

# ================ Helper Functions ================

def extract_domain(url):
    """Extract the domain name from a URL"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

async def fetch_url(url, timeout=30, max_retries=3, custom_headers=None, cookies=None):
    """Fetch content from a URL with advanced error handling and retries"""
    
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
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }
    
    # Update headers with custom headers if provided
    if custom_headers:
        headers.update(custom_headers)
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            async with aiohttp.ClientSession(cookies=cookies) as session:
                async with session.get(url, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:  # Too Many Requests
                        retry_delay = min(2 ** retry_count, 60)  # Exponential backoff with max 60 seconds
                        logger.warning(f"Rate limited on {url}, retrying in {retry_delay} seconds")
                        await asyncio.sleep(retry_delay)
                    elif response.status == 403:  # Forbidden (likely blocked)
                        logger.warning(f"Access forbidden to {url}, may be blocking scraping")
                        # Change user agent for next retry
                        headers['User-Agent'] = random.choice(user_agents)
                        await asyncio.sleep(5)
                    elif 300 <= response.status < 400:  # Handle redirects manually if needed
                        location = response.headers.get('Location')
                        if location:
                            logger.info(f"Following redirect from {url} to {location}")
                            return await fetch_url(location, timeout, max_retries - retry_count, custom_headers, cookies)
                        else:
                            logger.warning(f"Redirect without location from {url}")
                            return None
                    else:
                        logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        if retry_count + 1 < max_retries:
                            await asyncio.sleep(2)
                        else:
                            return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            await asyncio.sleep(5)
        except aiohttp.ClientError as e:
            logger.error(f"Client error fetching {url}: {str(e)}")
            await asyncio.sleep(3)
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            await asyncio.sleep(3)
        
        retry_count += 1
    
    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
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
    """Extract company name, description, and other metadata from HTML"""
    if not html_content:
        return {"name": "Unknown", "description": ""}
    
    result = {"name": "Unknown", "description": "", "keywords": [], "social_links": {}}
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract from JSON-LD structured data (if available)
        json_ld_data = extract_json_ld(soup)
        if json_ld_data:
            if 'Organization' in str(json_ld_data):
                for item in json_ld_data if isinstance(json_ld_data, list) else [json_ld_data]:
                    if item.get('@type') in ['Organization', 'Corporation', 'Company', 'LocalBusiness']:
                        result["name"] = item.get('name', result["name"])
                        result["description"] = item.get('description', result["description"])
                        if 'sameAs' in item and isinstance(item['sameAs'], list):
                            for social in item['sameAs']:
                                domain = extract_domain(social)
                                if domain in ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']:
                                    result["social_links"][domain] = social
        
        # Extract company name from various sources if not already found
        if result["name"] == "Unknown":
            # Try organization schema
            org_name = soup.find('meta', property='og:site_name')
            if org_name and org_name.get('content'):
                result["name"] = org_name['content']
                
            # Try title
            if result["name"] == "Unknown" and soup.title:
                title = soup.title.string
                # Clean title
                if title:
                    # Remove common title patterns
                    title = re.sub(r'(\s*\|.*$|\s*-.*$|\s*–.*$|\s*:.*$)', '', title).strip()
                    result["name"] = title
            
            # Try common header elements
            if result["name"] == "Unknown":
                for selector in ['header h1', 'header .logo', '#header .logo', '.header .logo', '.site-title', '.brand', '.logo']:
                    name_elem = soup.select_one(selector)
                    if name_elem:
                        if name_elem.has_attr('alt'):
                            result["name"] = name_elem['alt']
                            break
                        elif name_elem.has_attr('title'):
                            result["name"] = name_elem['title']
                            break
                        elif name_elem.get_text().strip():
                            result["name"] = name_elem.get_text().strip()
                            break
        
        # Extract description if not already found
        if not result["description"]:
            # Try meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                result["description"] = meta_desc['content']
            
            # Try OpenGraph description
            if not result["description"]:
                og_desc = soup.find('meta', property='og:description')
                if og_desc and og_desc.get('content'):
                    result["description"] = og_desc['content']
            
            # Try Twitter description
            if not result["description"]:
                twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
                if twitter_desc and twitter_desc.get('content'):
                    result["description"] = twitter_desc['content']
                    
            # Try the first paragraph in the main content
            if not result["description"]:
                for selector in ['main p', '#content p', '.content p', 'article p', '.about-us p', '.about p', '.company-info p']:
                    p_elems = soup.select(selector)
                    if p_elems:
                        desc = p_elems[0].get_text().strip()
                        if len(desc) > 50:  # Only use if it's substantial
                            result["description"] = desc
                            break
        
        # Extract keywords
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta and keywords_meta.get('content'):
            result["keywords"] = [k.strip() for k in keywords_meta['content'].split(',')]
            
        # Find social media links
        if not result["social_links"]:
            social_patterns = {
                'facebook.com': re.compile(r'facebook\.com/([^/"\']+)'),
                'twitter.com': re.compile(r'twitter\.com/([^/"\']+)'),
                'linkedin.com': re.compile(r'linkedin\.com/(?:company|in)/([^/"\']+)'),
                'instagram.com': re.compile(r'instagram\.com/([^/"\']+)')
            }
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                for domain, pattern in social_patterns.items():
                    if domain in href:
                        result["social_links"][domain] = href
        
        # Clean and truncate results
        if result["name"] != "Unknown":
            result["name"] = result["name"][:100]  # Limit name length
        
        if result["description"]:
            result["description"] = result["description"][:500]  # Limit description length
            
    except Exception as e:
        logger.error(f"Error extracting company info: {str(e)}")
    
    return result

def extract_json_ld(soup):
    """Extract and parse JSON-LD data from HTML"""
    try:
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                return data
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception as e:
        logger.error(f"Error extracting JSON-LD: {str(e)}")
    return None

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

# ================ Plugins ================
class ScrapingSessionManager:
    """Manages sessions for web scraping to maintain cookies and state"""
    
    def __init__(self):
        self.sessions = {}
        self.cookies = {}
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
        ]
    
    def get_domain_key(self, url):
        """Get a domain key for storing session data"""
        parsed_url = urlparse(url)
        return parsed_url.netloc
    
    async def fetch_with_session(self, url, timeout=30, max_retries=3, custom_headers=None):
        """Fetch URL content while maintaining session state"""
        domain = self.get_domain_key(url)
        
        # Create headers with the domain's user agent or pick a new one
        headers = {
            'User-Agent': self.sessions.get(domain, {}).get('user_agent', random.choice(self.user_agents)),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        if custom_headers:
            headers.update(custom_headers)
        
        # Get domain cookies or initialize empty
        cookies = self.cookies.get(domain, {})
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Create a session or use an existing one
                if domain not in self.sessions:
                    self.sessions[domain] = {
                        'user_agent': headers['User-Agent'],
                        'last_used': time.time()
                    }
                
                async with aiohttp.ClientSession(cookies=cookies) as session:
                    async with session.get(url, headers=headers, timeout=timeout) as response:
                        # Update cookies
                        if response.cookies:
                            for key, cookie in response.cookies.items():
                                cookies[key] = cookie.value
                            self.cookies[domain] = cookies
                        
                        if response.status == 200:
                            # Update session data
                            self.sessions[domain]['last_used'] = time.time()
                            return await response.text()
                        elif response.status == 429:  # Too Many Requests
                            retry_delay = min(2 ** retry_count, 60)
                            logger.warning(f"Rate limited on {url}, retrying in {retry_delay} seconds")
                            await asyncio.sleep(retry_delay)
                        elif response.status == 403:  # Forbidden (likely blocked)
                            logger.warning(f"Access forbidden to {url}, rotating user agent")
                            # Rotate user agent
                            headers['User-Agent'] = random.choice(self.user_agents)
                            self.sessions[domain]['user_agent'] = headers['User-Agent']
                            await asyncio.sleep(5)
                        else:
                            logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                            if retry_count + 1 < max_retries:
                                await asyncio.sleep(2)
                            else:
                                return None
            except Exception as e:
                logger.error(f"Error in fetch_with_session for {url}: {str(e)}")
                await asyncio.sleep(2)
            
            retry_count += 1
        
        return None
    
    def clean_old_sessions(self, max_age=3600):
        """Clean up sessions older than max_age seconds"""
        current_time = time.time()
        domains_to_remove = []
        
        for domain, session_data in self.sessions.items():
            if current_time - session_data['last_used'] > max_age:
                domains_to_remove.append(domain)
        
        for domain in domains_to_remove:
            self.sessions.pop(domain, None)
            self.cookies.pop(domain, None)
class WebScraperPlugin:
    """Plugin for web scraping operations"""
    
    @kernel_function(description="Scrape a webpage and return its content")
    async def scrape_webpage(self, url: str) -> str:
        """Scrape a webpage and return its content"""
        html_content = await fetch_url(url)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            # Remove extra whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text[:8000]  # Truncate to avoid token limits
        return "Failed to scrape the webpage."
    
    @kernel_function(description="Extract all URLs from a webpage")
    async def extract_urls(self, url: str) -> str:
        """Extract all URLs from a webpage"""
        html_content = await fetch_url(url)
        if not html_content:
            return json.dumps({"urls": []})
        
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
    
    @kernel_function(description="Extract product information from a webpage")
    async def extract_products(self, url: str) -> str:
        """Extract product information from a webpage with improved detection and OpenAI fallback"""
        try:
            html_content = await fetch_url(url)
            if not html_content:
                logger.warning(f"Failed to fetch content from {url}")
                return json.dumps({"products": []})
            
            # First, try to extract structured data (JSON-LD, microdata)
            products = self.extract_structured_product_data(html_content)
            
            # If structured data extraction found products, return them
            if products:
                logger.info(f"Extracted {len(products)} products using structured data from {url}")
                return json.dumps({"products": products})
            
            # Otherwise use traditional HTML parsing
            products = self.extract_products_from_html(html_content, url)
            
            # If still no products found, use OpenAI to assist
            if not products:
                logger.info(f"Traditional extraction failed, using OpenAI for {url}")
                extraction_goal = """
                Find all products on this page, including:
                - Product name
                - Price (as a numerical value without currency symbols)
                - Description (short if available)
                - Features (as an array of strings)
                - Any URLs for each product (or use the main URL if not available)
                
                Return as an array of product objects.
                """
                
                openai_client = AzureOpenAI(
                    api_version=AZURE_OPENAI_API_VERSION,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_key=AZURE_OPENAI_API_KEY
                )
                
                openai_result = await extract_with_openai(html_content, extraction_goal, openai_client)
                
                if 'products' in openai_result and isinstance(openai_result['products'], list):
                    products = openai_result['products']
                    # Ensure proper structure and fields
                    for product in products:
                        if 'id' not in product:
                            product['id'] = str(uuid.uuid4())[:8]
                        if 'url' not in product or not product['url']:
                            product['url'] = url
                        if 'features' not in product:
                            product['features'] = []
                        if 'last_updated' not in product:
                            product['last_updated'] = datetime.now().strftime("%Y-%m-%d")
                
            # Ensure we found at least some products
            if not products:
                logger.warning(f"No products found on {url} using any method")
                
                # Last resort: create a default product based on page title
                try:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    title = soup.title.string if soup.title else "Unknown Product"
                    products = [{
                        'id': str(uuid.uuid4())[:8],
                        'name': title,
                        'description': "Product information extracted from page",
                        'url': url,
                        'price': 0.0,
                        'features': [],
                        'last_updated': datetime.now().strftime("%Y-%m-%d")
                    }]
                except Exception as e:
                    logger.error(f"Error creating default product: {str(e)}")
                    products = []
            
            logger.info(f"Extracted {len(products)} products from {url}")
            return json.dumps({"products": products})
        
        except Exception as e:
            logger.error(f"Error extracting products from {url}: {str(e)}")
            return json.dumps({"products": []})

    def extract_structured_product_data(self, html_content):
        """Extract product data from structured data in the page"""
        products = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract from JSON-LD structured data
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    
                    # Handle different JSON-LD structures
                    if isinstance(data, list):
                        items = data
                    elif '@graph' in data:
                        items = data['@graph']
                    else:
                        items = [data]
                    
                    for item in items:
                        # Check for Product type
                        if item.get('@type') == 'Product':
                            product = {
                                'id': str(uuid.uuid4())[:8],
                                'name': item.get('name', 'Unknown Product'),
                                'description': item.get('description', ''),
                                'url': item.get('url', ''),
                                'features': []
                            }
                            
                            # Extract price from offers
                            if 'offers' in item:
                                offers = item['offers'] if isinstance(item['offers'], list) else [item['offers']]
                                for offer in offers:
                                    if 'price' in offer:
                                        try:
                                            product['price'] = float(offer['price'])
                                            break
                                        except (ValueError, TypeError):
                                            pass
                            
                            # If no price found
                            if 'price' not in product:
                                product['price'] = 0.0
                            
                            # Add product features
                            if 'additionalProperty' in item:
                                properties = item['additionalProperty']
                                for prop in properties if isinstance(properties, list) else [properties]:
                                    if 'name' in prop and 'value' in prop:
                                        product['features'].append(f"{prop['name']}: {prop['value']}")
                            
                            product['last_updated'] = datetime.now().strftime("%Y-%m-%d")
                            products.append(product)
                        
                        # Check for ItemList type
                        elif item.get('@type') == 'ItemList' and 'itemListElement' in item:
                            for element in item['itemListElement']:
                                if isinstance(element, dict) and element.get('item', {}).get('@type') == 'Product':
                                    product_item = element.get('item', {})
                                    product = {
                                        'id': str(uuid.uuid4())[:8],
                                        'name': product_item.get('name', 'Unknown Product'),
                                        'description': product_item.get('description', ''),
                                        'url': product_item.get('url', ''),
                                        'features': [],
                                        'last_updated': datetime.now().strftime("%Y-%m-%d")
                                    }
                                    
                                    # Extract price
                                    if 'offers' in product_item:
                                        offers = product_item['offers'] if isinstance(product_item['offers'], list) else [product_item['offers']]
                                        for offer in offers:
                                            if 'price' in offer:
                                                try:
                                                    product['price'] = float(offer['price'])
                                                    break
                                                except (ValueError, TypeError):
                                                    pass
                                    
                                    if 'price' not in product:
                                        product['price'] = 0.0
                                    
                                    products.append(product)
                except Exception as e:
                    logger.warning(f"Error parsing JSON-LD script: {str(e)}")
            
            # Extract from microdata
            microdata_products = soup.find_all(itemtype=re.compile(r'schema.org/Product'))
            for product_elem in microdata_products:
                try:
                    name_elem = product_elem.find(itemprop='name')
                    name = name_elem.get_text().strip() if name_elem else 'Unknown Product'
                    
                    description_elem = product_elem.find(itemprop='description')
                    description = description_elem.get_text().strip() if description_elem else ''
                    
                    url_elem = product_elem.find(itemprop='url')
                    url = url_elem['href'] if url_elem and url_elem.has_attr('href') else ''
                    
                    # Extract price
                    price = 0.0
                    price_elem = product_elem.find(itemprop='price')
                    if price_elem:
                        try:
                            price = float(re.sub(r'[^\d.]', '', price_elem.get_text()))
                        except (ValueError, TypeError):
                            pass
                    
                    # Extract features
                    features = []
                    feature_elems = product_elem.find_all(itemprop='additionalProperty')
                    for feature in feature_elems:
                        name_elem = feature.find(itemprop='name')
                        value_elem = feature.find(itemprop='value')
                        if name_elem and value_elem:
                            features.append(f"{name_elem.get_text()}: {value_elem.get_text()}")
                    
                    products.append({
                        'id': str(uuid.uuid4())[:8],
                        'name': name,
                        'description': description,
                        'url': url,
                        'price': price,
                        'features': features,
                        'last_updated': datetime.now().strftime("%Y-%m-%d")
                    })
                except Exception as e:
                    logger.warning(f"Error extracting microdata product: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error extracting structured product data: {str(e)}")
        
        return products

    def extract_products_from_html(self, html_content, base_url):
        """Extract products from HTML using heuristics"""
        products = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Common product container patterns
            product_selectors = [
                '.product', '.item', '.product-item', '.product-card', '.product-container',
                '[class*="product"]', '[class*="item"]', '[class*="card"]',
                '.listing', '.search-result', '[data-product-id]', '[data-item-id]',
                'li.item', 'div.item', 'article.product', '.grid-item'
            ]
            
            # Find product containers
            product_elements = []
            for selector in product_selectors:
                elements = soup.select(selector)
                product_elements.extend(elements)
            
            # Deduplicate elements (some may match multiple selectors)
            unique_elements = []
            for element in product_elements:
                if element not in unique_elements:
                    unique_elements.append(element)
            
            # Process each product element
            for element in unique_elements[:30]:  # Limit to 30 products
                # Extract product name
                name = None
                name_selectors = [
                    '[class*="title"]', '[class*="name"]', 'h1', 'h2', 'h3', 'h4', 'h5',
                    '[class*="product-name"]', '[class*="product-title"]'
                ]
                
                for selector in name_selectors:
                    name_elem = element.select_one(selector)
                    if name_elem and name_elem.get_text().strip():
                        name = name_elem.get_text().strip()
                        break
                
                # Skip if no name found - crucial for a product
                if not name:
                    continue
                
                # Extract price
                price = 0.0
                price_selectors = [
                    '[class*="price"]', '[data-price]', '.price', '.amount',
                    '[class*="cost"]', '[class*="amount"]'
                ]
                
                price_patterns = [
                    r'(\$|€|£|USD|EUR|GBP)?\s*(\d+(?:[.,]\d{1,2})?)',
                    r'(\d+(?:[.,]\d{1,2})?)\s*(\$|€|£|USD|EUR|GBP)',
                ]
                
                # Try to find price by selectors
                for selector in price_selectors:
                    price_elem = element.select_one(selector)
                    if price_elem:
                        price_text = price_elem.get_text().strip()
                        for pattern in price_patterns:
                            match = re.search(pattern, price_text)
                            if match:
                                try:
                                    # Extract the numeric part of the price
                                    price_str = match.group(1) if match.group(1) and match.group(1)[0].isdigit() else match.group(2)
                                    price = float(price_str.replace(',', '.'))
                                    break
                                except (ValueError, TypeError):
                                    pass
                    
                    if price > 0:
                        break
                
                # Extract description
                description = ""
                desc_selectors = [
                    '[class*="desc"]', '[class*="info"]', '[class*="summary"]', 'p', 
                    '[class*="detail"]', '[class*="text"]'
                ]
                
                for selector in desc_selectors:
                    desc_elem = element.select_one(selector)
                    if desc_elem and desc_elem.get_text().strip():
                        description = desc_elem.get_text().strip()
                        # Don't break here - try to find the longest description
                        if len(description) > 30:  # Skip very short descriptions
                            break
                
                # Extract URL
                url = ""
                link_elem = element.find('a')
                if link_elem and link_elem.has_attr('href'):
                    href = link_elem['href']
                    # Convert relative URLs to absolute
                    if not href.startswith(('http://', 'https://')):
                        if href.startswith('/'):
                            parsed_base = urlparse(base_url)
                            url = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
                        else:
                            url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                    else:
                        url = href
                
                # If URL not found, use the base URL
                if not url:
                    url = base_url
                
                # Extract features
                features = []
                feature_elems = element.select('ul li, ol li, [class*="feature"] li, [class*="spec"] li')
                for feature in feature_elems:
                    feature_text = feature.get_text().strip()
                    if feature_text:
                        features.append(feature_text)
                
                # Create product object
                product = {
                    'id': str(uuid.uuid4())[:8],
                    'name': name[:150],  # Limit name length
                    'price': price,
                    'description': description[:500],  # Limit description length
                    'url': url,
                    'features': features[:10],  # Limit to 10 features
                    'last_updated': datetime.now().strftime("%Y-%m-%d")
                }
                
                products.append(product)
            
        except Exception as e:
            logger.error(f"Error extracting products from HTML: {str(e)}")
        
        return products

class WebSearchPlugin:
    """Plugin for web search operations"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    @kernel_function(description="Search the web for information about a topic")
    async def search(self, query: str, num_results: int = 5) -> str:
        """Search the web using OpenAI's web browsing capability"""
        messages = [
            {"role": "system", "content": "You are a search assistant that helps users find information from the web. Provide a JSON array of search results with URL, title, and a brief summary for each."},
            {"role": "user", "content": f"Search for information about: {query}. Return results as JSON."}
        ]
        
        try:
            # Fixed: use model instead of deployment_id
            response = self.openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT,  # Correctly use model parameter with deployment name
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return json.dumps({"results": []})
    
    @kernel_function(description="Get news articles about a company or topic")
    async def get_news(self, query: str, days: int = 30) -> str:
        """Get recent news about a company or topic"""
        messages = [
            {"role": "system", "content": "You are a news search assistant that finds recent news articles about a topic or company. Provide a JSON array of news results with URL, title, date, and a brief summary."},
            {"role": "user", "content": f"Find news articles from the past {days} days about: {query}. Return results as JSON."}
        ]
        
        try:
            # Fixed: use model instead of deployment_id
            response = self.openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT,  # Correctly use model parameter with deployment name
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during news search: {str(e)}")
            return json.dumps({"articles": []})

class MarketTrendsPlugin:
    """Plugin for market trends analysis"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    @kernel_function(description="Analyze market trends and sentiment for an industry or company")
    async def analyze_market_trends(self, company: str, industry: str) -> str:
        """Analyze market trends for a company or industry"""
        messages = [
            {"role": "system", "content": "You are a market research analyst that identifies trends and sentiment. Provide a JSON object with trends, sentiment score, and key points."},
            {"role": "user", "content": f"Analyze current market trends and sentiment for {company} in the {industry} industry. Focus on the past 30 days. Return a detailed JSON with trends, sentiment_score, and key_points."}
        ]
        
        try:
            # Fixed: use model instead of deployment_id
            response = self.openai_client.chat.completions.create(
                model=CHAT_DEPLOYMENT,  # Correctly use model parameter with deployment name
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
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
        
        # Add plugins to the kernel
        self.web_scraper_plugin = kernel.add_plugin(WebScraperPlugin(), "WebScraper")
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
            You are an expert competitor intelligence analyst specialized in identifying top competitors for a company.
            Given a company URL, identify its top competitors by:
            1. Understanding the company's products and services
            2. Researching the industry and market
            3. Finding similar companies in the same space
            For each competitor, provide the company name, website URL, and a brief description.
            
            ALWAYS return your response in valid JSON format. Use the following structure:
            {
                "competitors": [
                    {
                        "name": "Competitor Name",
                        "url": "https://competitor-website.com",
                        "similarity_score": 0.9,
                        "description": "Brief description of competitor"
                    }
                ]
            }
            """,
            plugins=[self.web_scraper_plugin, self.web_search_plugin],
        )
    
    async def find_competitors(self, company_url: str, count: int = 5) -> List[CompetitorInfo]:
        """Find top competitors for the given company URL"""
        try:
            # First, get information about the company itself
            content = await fetch_url(company_url)
            company_info = extract_company_info(content)
            company_name = company_info.get("name", extract_domain(company_url))
            company_description = company_info.get("description", "")
            
            # Create query for the agent
            task = f"Find the top {count} competitors for {company_name} (URL: {company_url}). If you don't have enough information about the company, first research it using the WebScraper plugin. Then use the WebSearch plugin to find competitors. Return a JSON array of competitors with name, url, similarity_score (0-1), and a brief description for each."
            
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
            
            return competitors[:count]
        
        except Exception as e:
            logger.error(f"Error in CompetitorFinderAgent: {str(e)}")
            return []

class ProductScraperAgent:
    """Agent to extract product and pricing information from competitor websites"""
    
    def __init__(self, kernel, openai_client):
        self.kernel = kernel
        self.openai_client = openai_client
        
        # Add plugins to the kernel
        self.web_scraper_plugin = kernel.add_plugin(WebScraperPlugin(), "WebScraper")
        
        # Create the agent
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
            You are an expert product researcher specialized in extracting detailed product information from websites.
            Given a company website, identify their key products or services, including:
            1. Product names
            2. Pricing information
            3. Key features and descriptions
            4. URLs for each product page
            
            ALWAYS return your response in valid JSON format. Use the following structure:
            {
                "products": [
                    {
                        "id": "unique-id",
                        "name": "Product Name",
                        "price": 99.99,
                        "description": "Product description",
                        "url": "https://product-url.com",
                        "features": ["Feature 1", "Feature 2"],
                        "last_updated": "2025-05-16"
                    }
                ]
            }
            """,
            plugins=[self.web_scraper_plugin],
        )
    
    async def extract_products(self, company_url: str) -> List[ProductInfo]:
        """Extract products from a company website"""
        try:
            # First, scrape the main page to look for product links
            task = f"Extract product information from {company_url}. First use the WebScraper plugin to analyze the homepage and find product links. Then extract detailed product information from each link. Return a JSON array of products with id, name, price, description, url, features, and last_updated fields."
            
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
            
            for prod in products_data:
                if isinstance(prod, dict) and "name" in prod:
                    # Generate a product ID if not present
                    if "id" not in prod:
                        prod["id"] = str(uuid.uuid4())[:8]
                    
                    # Parse price to float if possible
                    price = 0.0
                    try:
                        price_str = str(prod.get("price", "0")).replace('$', '').replace('€', '').replace('£', '')
                        price = float(price_str)
                    except (ValueError, TypeError):
                        price = 0.0
                    
                    # Set last_updated to current time if not present
                    if "last_updated" not in prod:
                        prod["last_updated"] = datetime.now().strftime("%Y-%m-%d")
                    
                    products.append(ProductInfo(
                        id=prod.get("id", str(uuid.uuid4())[:8]),
                        name=prod.get("name", "Unknown"),
                        price=price,
                        description=prod.get("description", ""),
                        url=prod.get("url", company_url),
                        features=prod.get("features", []),
                        last_updated=prod.get("last_updated", datetime.now().strftime("%Y-%m-%d"))
                    ))
            
            return products
        
        except Exception as e:
            logger.error(f"Error in ProductScraperAgent: {str(e)}")
            return []

class MarketSentimentAgent:
    """Agent to analyze market sentiment and trends"""
    
    def __init__(self, kernel, openai_client):
        self.kernel = kernel
        self.openai_client = openai_client
        
        # Add plugins to the kernel
        self.web_search_plugin = kernel.add_plugin(WebSearchPlugin(openai_client), "WebSearch")
        self.market_trends_plugin = kernel.add_plugin(MarketTrendsPlugin(openai_client), "MarketTrends")
        
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
            You are an expert market intelligence analyst specialized in tracking sentiment and trends.
            Given a company or industry, analyze recent news, forum discussions, and social media to identify:
            1. Overall market sentiment as a float value between -1.0 (extremely negative) and 1.0 (extremely positive)
            2. Key trends and developments
            3. Emerging opportunities and threats
            4. User feedback and perceptions
            
            ALWAYS return your response in valid JSON format. Use the following structure:
            {
                "market_sentiment": {
                    "overall_sentiment": 0.7,
                    "key_points": ["Point 1", "Point 2"],
                    "sources": ["Source 1", "Source 2"]
                },
                "market_trends": [
                    {
                        "trend": "Trend Name",
                        "sources": ["Source 1", "Source 2"],
                        "relevance_score": 0.9,
                        "description": "Trend description"
                    }
                ]
            }
            
            IMPORTANT: Always use numerical values for overall_sentiment and relevance_score (not strings).
            """,
            plugins=[self.web_search_plugin, self.market_trends_plugin],
        )
    
    async def analyze_sentiment(self, company_name: str, industry: str) -> tuple[MarketSentiment, List[MarketTrend]]:
        """Analyze market sentiment and trends for a company and industry"""
        try:
            task = f"""
            Analyze market sentiment and trends for {company_name} in the {industry} industry over the past 30 days. 
            Use the WebSearch plugin to find recent news and discussions, then analyze the sentiment and identify key trends.
            
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
            You are an expert business intelligence analyst specialized in generating actionable insights and recommendations.
            Given comprehensive data about competitors, products, and market trends, you will:
            1. Identify key strategic insights about competitors
            2. Recommend product features or improvements
            3. Suggest pricing strategy adjustments
            4. Identify market opportunities and threats
            5. Provide specific, actionable next steps
            
            ALWAYS return your response in valid JSON format. Use the following structure:
            {
                "insights": [
                    {
                        "insight": "Insight description",
                        "relevance": 0.9,
                        "action_items": ["Action 1", "Action 2"]
                    }
                ],
                "recommendations": ["Recommendation 1", "Recommendation 2"]
            }
            
            Your insights should be data-driven, specific, and directly applicable to business decisions.
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
            
            # Create the agent task
            data_json = json.dumps({
                "company_name": company_name,
                "competitors": competitors_json,
                "products": products_json,
                "market_sentiment": sentiment_json,
                "market_trends": trends_json
            }, indent=2)
            
            task = f"""
            Generate strategic insights and recommendations based on the following competitor and market analysis:
            
            {data_json}
            
            Return a JSON object with:
            1. "insights": An array of insight objects, each with "insight", "relevance" (0-1), and "action_items" fields
            2. "recommendations": An array of specific, actionable recommendations
            
            Focus on practical, actionable insights that can drive business decisions.
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
                insights.append(InsightItem(
                    insight=insight_data.get("insight", ""),
                    relevance=float(insight_data.get("relevance", 0.5)),
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
    """Run the full competitor and market analysis workflow"""
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
        
        # Update to use deployment_name instead of ai_model_id
        kernel.add_service(
            AzureChatCompletion(
                service_id=service_id,
                deployment_name=CHAT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            )
        )
        
        # Update to use deployment_name instead of ai_model_id
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
        
        content = await fetch_url(url)
        company_info = extract_company_info(content)
        company_name = company_info.get("name", extract_domain(url))
        company_domain = extract_domain(url)
        
        # Get products for the target company
        target_products = await product_agent.extract_products(url)
        
        # Step 2: Find competitors
        tasks[task_id]["progress"] = 0.4
        tasks[task_id]["message"] = f"Identifying competitors for {company_name}"
        
        competitors = await competitor_agent.find_competitors(url, count=5)
        
        # Step 3: Extract products from competitors
        tasks[task_id]["progress"] = 0.5
        tasks[task_id]["message"] = "Analyzing competitor products"
        
        all_products = {company_name: target_products}
        for competitor in competitors:
            competitor_products = await product_agent.extract_products(competitor.url)
            all_products[competitor.name] = competitor_products
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(1)
        
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
        
        # Create the final report
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
        
        # Save the report
        report_filename = f"{company_domain}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(os.path.join(DATA_DIR, report_filename), 'w') as f:
            f.write(report.model_dump_json(indent=2))
        
        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 1.0
        tasks[task_id]["message"] = "Analysis completed successfully"
        tasks[task_id]["result"] = report
        
        return report
    
    except Exception as e:
        logger.error(f"Error in analysis workflow: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Analysis failed: {str(e)}"
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

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        message=task_info["message"],
        result=task_info["result"]
    )

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

# ================ Unit Tests ================

# These would normally be in a separate file, but are included here for completeness

async def test_competitor_finder_agent():
    """Test the CompetitorFinderAgent"""
    kernel = Kernel()
    
    # Update to use deployment_name instead of ai_model_id
    kernel.add_service(
        AzureChatCompletion(
            service_id="chat",
            deployment_name=CHAT_DEPLOYMENT,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
    )
    
    openai_client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    
    agent = CompetitorFinderAgent(kernel, openai_client)
    competitors = await agent.find_competitors("https://www.apple.com")
    
    assert len(competitors) > 0, "Failed to find any competitors"
    assert all(isinstance(comp, CompetitorInfo) for comp in competitors), "Invalid competitor information"
    
    print("CompetitorFinderAgent test passed")
    return competitors

async def test_product_scraper_agent():
    """Test the ProductScraperAgent"""
    kernel = Kernel()
    
    # Update to use deployment_name instead of ai_model_id
    kernel.add_service(
        AzureChatCompletion(
            service_id="chat",
            deployment_name=CHAT_DEPLOYMENT,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
    )
    
    openai_client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    
    agent = ProductScraperAgent(kernel, openai_client)
    products = await agent.extract_products("https://www.apple.com")
    
    assert len(products) > 0, "Failed to extract any products"
    assert all(isinstance(prod, ProductInfo) for prod in products), "Invalid product information"
    
    print("ProductScraperAgent test passed")
    return products

async def test_market_sentiment_agent():
    """Test the MarketSentimentAgent"""
    kernel = Kernel()
    
    # Update to use deployment_name instead of ai_model_id
    kernel.add_service(
        AzureChatCompletion(
            service_id="chat",
            deployment_name=CHAT_DEPLOYMENT,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
    )
    
    openai_client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    
    agent = MarketSentimentAgent(kernel, openai_client)
    sentiment, trends = await agent.analyze_sentiment("Apple", "Technology")
    
    assert isinstance(sentiment, MarketSentiment), "Invalid sentiment object"
    assert len(trends) > 0, "Failed to identify any market trends"
    
    print("MarketSentimentAgent test passed")
    return sentiment, trends

async def test_insight_generator_agent():
    """Test the InsightGeneratorAgent"""
    kernel = Kernel()
    
    # Update to use deployment_name instead of ai_model_id
    kernel.add_service(
        AzureChatCompletion(
            service_id="chat",
            deployment_name=CHAT_DEPLOYMENT,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
    )
    
    openai_client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    
    # Create sample data
    competitors = [
        CompetitorInfo(name="Samsung", url="https://www.samsung.com", similarity_score=0.9, description="Consumer electronics company"),
        CompetitorInfo(name="Google", url="https://www.google.com", similarity_score=0.8, description="Technology company")
    ]
    
    products = {
        "Apple": [
            ProductInfo(id="1", name="iPhone", price=999.0, description="Smartphone", url="https://www.apple.com/iphone", features=["A15 chip", "Pro camera"], last_updated=datetime.now().isoformat())
        ],
        "Samsung": [
            ProductInfo(id="2", name="Galaxy S", price=899.0, description="Smartphone", url="https://www.samsung.com/galaxy", features=["Snapdragon chip", "Pro camera"], last_updated=datetime.now().isoformat())
        ]
    }
    
    sentiment = MarketSentiment(
        overall_sentiment=0.7,
        key_points=["Strong brand loyalty", "Innovation leadership"],
        sources=["News articles", "Social media"]
    )
    
    trends = [
        MarketTrend(
            trend="AI integration",
            sources=["Tech blogs", "Industry reports"],
            relevance_score=0.9,
            description="Integration of AI features in consumer products"
        )
    ]
    
    agent = InsightGeneratorAgent(kernel, openai_client)
    insights, recommendations = await agent.generate_insights("Apple", competitors, products, sentiment, trends)
    
    assert len(insights) > 0, "Failed to generate any insights"
    assert len(recommendations) > 0, "Failed to generate any recommendations"
    
    print("InsightGeneratorAgent test passed")
    return insights, recommendations

@app.get("/run-tests", include_in_schema=False)
async def run_tests():
    """Run unit tests for all agents"""
    try:
        competitors = await test_competitor_finder_agent()
        products = await test_product_scraper_agent()
        sentiment, trends = await test_market_sentiment_agent()
        insights, recommendations = await test_insight_generator_agent()
        
        return {
            "status": "success",
            "message": "All tests passed",
            "sample_data": {
                "competitors": [comp.model_dump() for comp in competitors[:2]],
                "products": [prod.model_dump() for prod in products[:2]],
                "sentiment": sentiment.model_dump(),
                "trends": [trend.model_dump() for trend in trends[:2]],
                "insights": [insight.model_dump() for insight in insights[:2]],
                "recommendations": recommendations[:2]
            }
        }
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return {
            "status": "failed",
            "message": f"Tests failed: {str(e)}"
        }

# ================ Main Entry Point ================
session_manager = ScrapingSessionManager()

# Schedule cleaning of old sessions
async def clean_sessions_periodically():
    while True:
        await asyncio.sleep(3600)  # Clean every hour
        session_manager.clean_old_sessions()

# Start the cleaning task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(clean_sessions_periodically())
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
