"""
core/tools/web_tools.py
Agentic web browsing/research capability using DuckDuckGo and BeautifulSoup.
"""

import asyncio
import logging
from typing import Optional

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger("Jarvis.WebTools")


async def web_search(query: str, max_results: int = 5) -> str:
    """Perform a web search using DuckDuckGo."""
    if DDGS is None:
        return "Error: duckduckgo-search package is not installed."

    try:
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await asyncio.to_thread(_search)
        
        if not results:
            return f"No results found for query: {query}"
        
        formatted = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "No Title")
            href = res.get("href", "No URL")
            body = res.get("body", "No description")
            formatted.append(f"{i}. {title}\nURL: {href}\nSummary: {body}\n")
            
        return "\n".join(formatted)
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return f"Search failed: {exc}"


async def web_scrape(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract readable text from a webpage."""
    if BeautifulSoup is None:
        return "Error: requests or beautifulsoup4 package is not installed."
        
    try:
        def _scrape():
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # Add timeout to avoid hanging indefinitely
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            # Use html.parser since lxml might not be fully reliable across systems
            # although duckduckgo search does install it
            soup = BeautifulSoup(resp.content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]) :
                script.decompose()
                
            # Extract text
            text = soup.get_text(separator="\n")
            
            # Collapse multiple spaces and newlines
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            return text

        text = await asyncio.to_thread(_scrape)
        
        if not text:
            return "Failed to extract readable text from the page."
            
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n...[Truncated, total {len(text)} chars]..."
            
        return text
    except requests.exceptions.Timeout:
        return f"Timeout while trying to reach: {url}"
    except requests.exceptions.RequestException as exc:
        return f"Failed to fetch {url}: {exc}"
    except Exception as exc:
        logger.error("Web scrape failed for %s: %s", url, exc)
        return f"Scraping failed: {exc}"
