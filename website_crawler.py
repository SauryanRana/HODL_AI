import asyncio
import json
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright, Playwright # Keep Playwright for potential specific TimeoutError
from bs4 import BeautifulSoup
import time # Keep for the main execution block if not fully async, but internal sleeps will be asyncio

# Helper function to check if a URL is within the same domain as the base URL
def is_within_domain(url, base_url):
    """Checks if the given URL is within the same domain as the base URL."""
    return urlparse(url).netloc == urlparse(base_url).netloc

# Function to extract visible text content from HTML
def extract_text_from_html(html_content, url_for_logging):
    """
    Extracts visible text from HTML content.
    Args:
        html_content (str): The HTML content of the page.
        url_for_logging (str): The URL of the page, for logging purposes.
    Returns:
        str: Cleaned, visible text content, or None if an error occurs.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text
        text_parts = soup.find_all(string=True)
        
        # Filter out unwanted elements and strip whitespace
        visible_text = []
        for t in text_parts:
            if t.parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]']:
                stripped_text = t.strip()
                if stripped_text: # Only add non-empty strings
                    visible_text.append(stripped_text)
        
        # Join the text parts and clean up multiple newlines
        cleaned_text = "\n".join(visible_text)
        cleaned_text = "\n".join([line for line in cleaned_text.splitlines() if line.strip()])
        
        return cleaned_text
    except Exception as e:
        print(f"Error extracting text from HTML of {url_for_logging}: {e}")
        return None

# Main crawling function using Playwright
async def crawl_website_playwright(start_url, disallowed_path_prefixes=None):
    """
    Crawls a website starting from start_url, extracts text content,
    and finds internal links.
    Args:
        start_url (str): The URL to start crawling from.
        disallowed_path_prefixes (list, optional): A list of path prefixes to ignore. 
                                                  Defaults to None.
    Returns:
        dict: A dictionary where keys are URLs and values are their extracted text content.
    """
    if disallowed_path_prefixes is None:
        disallowed_path_prefixes = []
        
    extracted_data = {}
    
    async with async_playwright() as p:
        # Launch the browser (Chromium is a good default)
        browser = await p.chromium.launch()
        
        # Create a new browser context with a common user-agent
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()

        visited = set()
        queue = [start_url]
        parsed_start_url = urlparse(start_url)
        base_domain = parsed_start_url.netloc # Store base domain for checking

        # Define common file extensions to ignore
        ignored_extensions = (
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', 
            '.zip', '.rar', '.exe', '.dmg', '.tar.gz', '.iso',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv',
            '.mp3', '.wav', '.ogg', '.aac', '.flac',
            '.css', '.js' 
        )

        while queue:
            current_url = queue.pop(0)
            
            parsed_current_url_for_visit_check = urlparse(current_url)
            # Normalize URL for visited check (remove query and fragment)
            url_to_visit_key = parsed_current_url_for_visit_check._replace(query='', fragment='').geturl()

            if url_to_visit_key in visited:
                continue
            visited.add(url_to_visit_key) 
            
            print(f"Crawling: {current_url}")

            try:
                await page.goto(current_url, wait_until='domcontentloaded', timeout=60000)
                await page.wait_for_load_state("networkidle", timeout=30000) 
                html_content = await page.content()

                text_content = extract_text_from_html(html_content, current_url)
                if text_content:
                    extracted_data[current_url] = text_content

                soup_for_links = BeautifulSoup(html_content, 'html.parser')
                for link_tag in soup_for_links.find_all('a', href=True):
                    href = link_tag['href'].strip()
                    if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:') or href.startswith('javascript:'):
                        continue

                    # Check if href is an absolute link to a different domain BEFORE urljoin
                    parsed_href = urlparse(href)
                    if parsed_href.netloc and parsed_href.netloc != base_domain:
                        # print(f"Skipping absolute external link: {href}")
                        continue
                    
                    absolute_url = urljoin(current_url, href)
                    parsed_absolute_url = urlparse(absolute_url)

                    # Skip if it contains "undefined" as a path segment
                    if "undefined" in parsed_absolute_url.path.lower().split('/'):
                        # print(f"Skipping URL with 'undefined' segment: {absolute_url}")
                        continue

                    # Skip if path starts with a disallowed prefix
                    is_disallowed_path = False
                    for prefix in disallowed_path_prefixes:
                        if parsed_absolute_url.path.startswith(prefix):
                            is_disallowed_path = True
                            # print(f"Skipping disallowed path: {absolute_url}")
                            break
                    if is_disallowed_path:
                        continue
                    
                    # Normalize URL for queue and visited check (remove query and fragment)
                    normalized_url_for_queue = parsed_absolute_url._replace(query='', fragment='').geturl()

                    if is_within_domain(normalized_url_for_queue, start_url) and \
                       normalized_url_for_queue not in visited and \
                       normalized_url_for_queue not in queue and \
                       not normalized_url_for_queue.lower().endswith(ignored_extensions):
                        queue.append(normalized_url_for_queue)
            
            except Exception as e: 
                print(f"Error processing page {current_url}: {e}")
            
            await asyncio.sleep(0.5)
            
        await browser.close()
    return extracted_data

# Main execution block
if __name__ == '__main__':
    start_url = "https://hodltoken.net/"
    # Add any path prefixes you want to specifically ignore
    ignored_paths = ["/uk/", "/shop/"] # Example: ignore /uk/ and /shop/ paths
    
    print(f"Starting crawl for: {start_url}")
    print(f"Ignoring path prefixes: {ignored_paths}")
    
    website_data = asyncio.run(crawl_website_playwright(start_url, disallowed_path_prefixes=ignored_paths))

    output_filename = f"{urlparse(start_url).netloc}_data_playwright.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(website_data, f, ensure_ascii=False, indent=4)

    print(f"\nCrawling complete. Extracted data saved to {output_filename}")
