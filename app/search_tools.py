import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import urllib.parse
import re
import logging
import base64

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _clean_ddg_url(raw_url: str) -> str:
    """Extract destination URL from DuckDuckGo redirect link."""
    if not raw_url:
        return ""
    if "uddg=" in raw_url:
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
        return parsed.get("uddg", [raw_url])[0]
    return raw_url


async def _search_ddg_html_post(client: httpx.AsyncClient, query: str, max_results: int) -> List[Dict[str, str]]:
    """Try DuckDuckGo HTML endpoint via POST."""
    results = []
    try:
        resp = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": ""},
            headers={**HEADERS, "Referer": "https://html.duckduckgo.com/"}
        )
        if resp.status_code == 200 and "bots use DuckDuckGo too" not in resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            for div in soup.select(".result")[:max_results]:
                title_tag = div.select_one(".result__title a, .result__a")
                snippet_tag = div.select_one(".result__snippet")
                if title_tag:
                    url = _clean_ddg_url(title_tag.get("href", ""))
                    if url.startswith("http"):
                        results.append({
                            "title": title_tag.get_text(strip=True),
                            "url": url,
                            "snippet": snippet_tag.get_text(strip=True) if snippet_tag else ""
                        })
    except Exception as e:
        logger.debug("DDG HTML POST error: %s", e)
    return results


async def _search_ddg_html_get(client: httpx.AsyncClient, query: str, max_results: int) -> List[Dict[str, str]]:
    """Try DuckDuckGo HTML endpoint via GET."""
    results = []
    try:
        resp = await client.get(
            f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}",
            headers={**HEADERS, "Referer": "https://html.duckduckgo.com/"}
        )
        if resp.status_code == 200 and "bots use DuckDuckGo too" not in resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            for div in soup.select(".result")[:max_results]:
                title_tag = div.select_one(".result__title a, .result__a")
                snippet_tag = div.select_one(".result__snippet")
                if title_tag:
                    url = _clean_ddg_url(title_tag.get("href", ""))
                    if url.startswith("http"):
                        results.append({
                            "title": title_tag.get_text(strip=True),
                            "url": url,
                            "snippet": snippet_tag.get_text(strip=True) if snippet_tag else ""
                        })
    except Exception as e:
        logger.debug("DDG HTML GET error: %s", e)
    return results


async def _search_ddg_lite(client: httpx.AsyncClient, query: str, max_results: int) -> List[Dict[str, str]]:
    """Try DuckDuckGo Lite endpoint via POST and GET."""
    results = []
    # 1. Try POST
    try:
        resp = await client.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers={**HEADERS, "Referer": "https://lite.duckduckgo.com/"}
        )
        if resp.status_code == 200 and "bots use DuckDuckGo too" not in resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            for link_tag in soup.select(".result-link")[:max_results]:
                url = _clean_ddg_url(link_tag.get("href", ""))
                tr = link_tag.find_parent("tr")
                snippet = ""
                if tr:
                    next_tr = tr.find_next_sibling("tr")
                    if next_tr:
                        snippet = next_tr.get_text(strip=True)
                if url.startswith("http"):
                    results.append({
                        "title": link_tag.get_text(strip=True),
                        "url": url,
                        "snippet": snippet
                    })
            if results:
                return results
    except Exception as e:
        logger.debug("DDG Lite POST error: %s", e)

    # 2. Try GET
    try:
        resp = await client.get(
            f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}",
            headers={**HEADERS, "Referer": "https://lite.duckduckgo.com/"}
        )
        if resp.status_code == 200 and "bots use DuckDuckGo too" not in resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            for link_tag in soup.select(".result-link")[:max_results]:
                url = _clean_ddg_url(link_tag.get("href", ""))
                tr = link_tag.find_parent("tr")
                snippet = ""
                if tr:
                    next_tr = tr.find_next_sibling("tr")
                    if next_tr:
                        snippet = next_tr.get_text(strip=True)
                if url.startswith("http"):
                    results.append({
                        "title": link_tag.get_text(strip=True),
                        "url": url,
                        "snippet": snippet
                    })
    except Exception as e:
        logger.debug("DDG Lite GET error: %s", e)
    return results


async def _search_bing(client: httpx.AsyncClient, query: str, max_results: int) -> List[Dict[str, str]]:
    """Fallback search using Bing."""
    results = []
    try:
        resp = await client.get(
            f"https://www.bing.com/search?q={urllib.parse.quote(query)}",
            headers={**HEADERS, "Referer": "https://www.bing.com/"}
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for li in soup.select("li.b_algo")[:max_results]:
                a = li.select_one("h2 a")
                if not a or not a.get("href"):
                    continue
                href = a["href"]
                # Decode Bing redirect if applicable
                if "/ck/a?!" in href and "u=" in href:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    u = parsed.get("u", [""])[0]
                    if u.startswith("a1"):
                        try:
                            padding = "=" * (-len(u[2:]) % 4)
                            decoded = base64.b64decode(u[2:] + padding).decode("utf-8", errors="ignore")
                            if decoded.startswith("http"):
                                href = decoded
                        except Exception:
                            pass
                snippet_elem = li.select_one(".b_caption p, p")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                title = a.get_text(strip=True)
                if href.startswith("http") and "bing.com" not in href:
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet
                    })
    except Exception as e:
        logger.debug("Bing search error: %s", e)
    return results


async def _search_wikipedia(client: httpx.AsyncClient, query: str, max_results: int) -> List[Dict[str, str]]:
    """Fallback search using Wikipedia REST API (reliable, encyclopedic, zero blocking)."""
    results = []
    # Determine language
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", query))
    domains = ["ru.wikipedia.org", "en.wikipedia.org"] if has_cyrillic else ["en.wikipedia.org"]
    
    for domain in domains:
        try:
            wiki_headers = {"User-Agent": "FastSearchAgent/1.0 (web-assistant@mail.com)"}
            url = f"https://{domain}/w/rest.php/v1/search/page?q={urllib.parse.quote(query)}&limit={max_results}"
            resp = await client.get(url, headers=wiki_headers)
            if resp.status_code == 200:
                data = resp.json()
                for page in data.get("pages", []):
                    title = page.get("title", "")
                    key = page.get("key", "")
                    description = page.get("description", "") or page.get("excerpt", "")
                    # Clean html excerpt tags if present
                    description = re.sub(r"<[^>]+>", "", description)
                    page_url = f"https://{domain}/wiki/{urllib.parse.quote(key)}"
                    results.append({
                        "title": f"{title} - Wikipedia",
                        "url": page_url,
                        "snippet": description
                    })
                    if len(results) >= max_results:
                        return results
        except Exception as e:
            logger.debug("Wikipedia search error (%s): %s", domain, e)
    return results


async def search_duckduckgo(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search the web with automatic fallback across multiple engines:
    1. DuckDuckGo HTML POST
    2. DuckDuckGo HTML GET
    3. DuckDuckGo Lite (POST & GET)
    4. Bing Search
    5. Wikipedia Search
    Guarantees that valid web search results are found.
    """
    if not query or not query.strip():
        return []

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        # 1. DuckDuckGo HTML POST
        results = await _search_ddg_html_post(client, query, max_results)
        if results:
            return results

        # 2. DuckDuckGo HTML GET
        results = await _search_ddg_html_get(client, query, max_results)
        if results:
            return results

        # 3. DuckDuckGo Lite
        results = await _search_ddg_lite(client, query, max_results)
        if results:
            return results

        # 4. Fallback: Bing Search
        logger.info("DuckDuckGo was blocked or returned 0 results for '%s'; falling back to Bing Search", query)
        results = await _search_bing(client, query, max_results)
        if results:
            return results

        # 5. Fallback: Wikipedia Search
        logger.info("Bing returned 0 results for '%s'; falling back to Wikipedia Search", query)
        results = await _search_wikipedia(client, query, max_results)
        if results:
            return results

    return []


async def fetch_page_content(url: str, max_chars: int = 5000) -> str:
    """Fetch and extract main text content from a URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script, style, nav, footer tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()

            # Extract text from main content areas
            main_content = soup.select_one("main, article, .content, #content, .post-content, #bodyContent")
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)

            return text[:max_chars]
    except Exception as e:
        logger.warning("Error fetching %s: %s", url, e)
        return f"[Error fetching page: {e}]"
