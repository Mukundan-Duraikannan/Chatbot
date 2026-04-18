from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import json
   
def scrape_website(base_url, max_pages=100):
    visited = set()
    queue = deque([base_url])
    pages = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        while queue and len(pages) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            try:
                page.goto(url, timeout=90000)
                page.wait_for_timeout(9000)
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                title = url
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
                pages.append({
                    "url": url,
                    "title": title,
                    "content": text
                })
 
                print("Scraped:", url)
 
                links = soup.find_all("a", href=True)
 
                for link in links:
                    href = link.get("href")
 
                    if href:
                        full_url = urljoin(url, str(href))
                        parsed = urlparse(full_url)
 
                        if parsed.netloc == urlparse(base_url).netloc:
                            if full_url not in visited:
                                queue.append(full_url)
 
            except Exception as e:
                print("Failed:", url, e)
 
        browser.close()
 
    return pages
 
 
if __name__ == "__main__":
    data = scrape_website("https://cogniwide.com/")
    print("Total pages scraped:", len(data))
 
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)