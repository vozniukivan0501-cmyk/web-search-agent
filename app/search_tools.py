"""Web search tools using DuckDuckGo (no API key required) and web page scraping."""
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import re
import logging
import urllib.parse

logger = logging.getLogger(__name__)


async def search_duckduckgo(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search DuckDuckGo and return results with title, url, snippet."""
    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            # Use DuckDuckGo HTML search
            params = {"q": query, "kl": "us-en"}
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            for result_div in soup.select(".result")[:max_results]:
                title_tag = result_div.select_one(".result__title a, .result__a")
                snippet_tag = result_div.select_one(".result__snippet")
                
                if title_tag:
                    href = title_tag.get("href", "")
                    # DuckDuckGo wraps URLs in redirect links
                    if "uddg=" in href:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        url = parsed.get("uddg", [href])[0]
                    else:
                        url = href
                    
                    results.append({
                        "title": title_tag.get_text(strip=True),
                        "url": url,
                        "snippet": snippet_tag.get_text(strip=True) if snippet_tag else ""
                    })
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
    
    return results


async def fetch_page_content(url: str, max_chars: int = 5000) -> str:
    """Fetch and extract main text content from a URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script, style, nav, footer tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            
            # Extract text from main content areas
            main_content = soup.select_one("main, article, .content, #content, .post-content")
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)
            
            return text[:max_chars]
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return f"[Error fetching page: {e}]"
