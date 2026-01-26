import asyncio
import json
import os
import re
import time
from urllib.parse import urljoin, urlparse
from typing import List

import aiohttp
from bs4 import BeautifulSoup
from urllib import robotparser
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# ---------- Config ----------
USER_AGENT = "MyCrawlerBot/1.0 (+https://example.com)"
MAX_PAGES = 500
OUTPUT_JSON = "scraped_data.json"
# ----------------------------

# Normalize domains
def normalize_domain(url_or_netloc: str) -> str:
    if not url_or_netloc:
        return ""
    if "://" in url_or_netloc:
        netloc = urlparse(url_or_netloc).netloc.lower()
    else:
        netloc = url_or_netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc

# Safe folder names
def slugify_url(url: str) -> str:
    u = url.replace("://", "_").replace("/", "_").strip("_")
    u = re.sub(r"[^A-Za-z0-9_\-\.]", "_", u)
    return u[:200]

# Cleaning functions (pre-processing markdown)
def clean_markdown_empty_images(md: str) -> str:
    md = re.sub(r'!\s*\[\s*\]\s*\(\s*\)', ' ', md)
    md = re.sub(r'!\s*\[.*?\]\s*\(\s*\)', ' ', md)
    md = re.sub(r'!\s*!?\s*\]\(\s*\)', ' ', md)
    return md

def clean_text_bulk(text: str) -> str:
    if not text:
        return ""
    text = clean_markdown_empty_images(text)
    text = re.sub(r'\[([^\]]{1,200}?)\]\((?:[^)]*?)\)', r'\1', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)

    menu_words = [
        r"\bRetour\b", r"\bFermer le sous-menu\b", r"\bMenu\b",
        r"\bAccueil\b", r"\bA propos\b", r"\bDécouvrir\b",
        r"\bSe déplacer\b", r"\bTitres et tarifs\b", r"\bMe connecter\b",
        r"\bPanier\b", r"\bProfil\b", r"\bPréparer mon voyage\b"
    ]
    text = re.sub("|".join(menu_words), " ", text, flags=re.IGNORECASE)
    text = re.sub(r'[*•#>\-]{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Robots.txt check
def is_allowed_by_robots(base_url: str, user_agent: str, test_url: str) -> bool:
    try:
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, test_url)
    except Exception:
        return True

# Enhanced boilerplate removal function
def extract_main_content_from_text(text: str) -> str:
    """Remove headers, footers, navigation and other boilerplate content."""
    if not text or len(text) < 200:
        return text
    
    lines = text.split('\n')
    
    # Keywords typically found in headers/footers/navigation
    boilerplate_patterns = [
        r'^(home|accueil|inicio|start)',
        r'(contact|copyright|©|rights reserved)',
        r'(privacy|terms|legal|policy)',
        r'(subscribe|newsletter|follow us)',
        r'(all rights reserved|powered by)',
        r'(menu|navigation|breadcrumb)',
        r'(skip to|jump to|back to top)',
        r'(social media|facebook|twitter|instagram)',
        r'(cookie|consent|gdpr)',
        r'(login|sign in|register|account)'
    ]
    
    # Filter out lines that match boilerplate patterns
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip().lower()
        if len(line_stripped) < 10:  # Keep very short lines
            continue
        
        # Check if line matches any boilerplate pattern
        is_boilerplate = False
        for pattern in boilerplate_patterns:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                is_boilerplate = True
                break
        
        # Keep lines that are substantial content
        if not is_boilerplate and len(line_stripped) > 30:
            cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines).strip()
    
    # Return original if cleaning removed too much
    if len(cleaned_text) < len(text) * 0.3:
        return text
    
    return cleaned_text

# Validate URL function
async def validate_url(url: str) -> tuple[bool, str]:
    """Validate if URL is accessible and returns proper response."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=10, ssl=False) as response:
                if response.status in [200, 301, 302, 307, 308]:
                    return True, ""
                else:
                    return False, f"HTTP {response.status}: {response.reason}"
    except aiohttp.ClientError as e:
        return False, f"Connection error: {str(e)}"
    except asyncio.TimeoutError:
        return False, "Timeout: Server not responding"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

# Main crawl function
async def crawl_site(max_attempts: int = 5):
    """Main scraping function with retry logic for domain validation."""
    attempts = 0
    
    while attempts < max_attempts:
        base_url = input("Enter a website (ex: https://www.tcl.fr): ").strip()
        
        if not base_url:
            attempts += 1
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"❌ No URL provided. {remaining} attempts remaining.")
                continue
            else:
                print("❌ Maximum attempts reached. Exiting.")
                return
        
        # Ensure proper URL format
        if not base_url.startswith(("http://", "https://")):
            base_url = "https://" + base_url
        
        print(f"🔍 Validating URL: {base_url}")
        
        # Validate the URL
        is_valid, error_msg = await validate_url(base_url)
        if is_valid:
            print("✅ URL validation successful!")
            break
        else:
            attempts += 1
            remaining = max_attempts - attempts
            
            print(f"❌ Error accessing website: {error_msg}")
            print("💡 Please check:")
            print("   - The domain name is correct")
            print("   - The website is online and accessible")
            print("   - You have internet connection")
            
            if remaining > 0:
                print(f"🔄 {remaining} attempts remaining. Please try again.")
                print("---")
            else:
                print("❌ Maximum attempts reached. Exiting.")
                return
    
    # Continue with the rest of the scraping logic
    normalized_domain = normalize_domain(base_url)
    print(f"✅ URL validation successful")
    print(f"🚀 Starting crawl for domain: {normalized_domain}")

    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        print(f"🔍 Crawling main page: {base_url}")
        main_result = await crawler.arun(url=base_url, config=run_config)

        if not main_result.success:
            print(f"❌ Failed to crawl main page: {main_result.error_message}")
            print("💡 Possible causes:")
            print("   - Website blocks automated crawlers")
            print("   - Network connectivity issues")
            print("   - Website requires authentication")
            return

        raw_internal = main_result.links.get("internal", []) or []
        print(f"📄 Found {len(raw_internal)} internal link entries on main page.")
        
        # Handle case where no internal links are found
        if len(raw_internal) == 0:
            print("⚠️  Warning: No internal links found on main page.")
            print("   Will proceed with scraping only the main page.")
            to_crawl = [base_url]  # Still crawl the main page

        to_crawl = []
        for entry in raw_internal:
            href = entry.get("href") if isinstance(entry, dict) else entry
            if not href:
                continue

            full = urljoin(base_url, href)
            if full.startswith("//"):
                full = "https:" + full

            if normalize_domain(full) != normalized_domain:
                continue
            if not is_allowed_by_robots(base_url, USER_AGENT, full):
                continue

            to_crawl.append(full)

        # Deduplicate and limit
        seen = set()
        filtered = []
        for u in to_crawl:
            if u not in seen:
                seen.add(u)
                filtered.append(u)
            if len(filtered) >= MAX_PAGES:
                break

        print(f"📌 Pages to crawl: {len(filtered)}")
        
        # Handle edge cases
        if len(filtered) == 0:
            print("⚠️  No valid pages to crawl after filtering.")
            print("   Will proceed with scraping only the main page.")
            filtered = [base_url]

        # Crawl pages in chunks with error handling
        chunk_size = 30
        results = []
        successful_crawls = 0
        failed_crawls = 0
                
        for i in range(0, len(filtered), chunk_size):
            chunk = filtered[i:i + chunk_size]
            print(f"⏳ Crawling chunk {i//chunk_size + 1} ({len(chunk)} pages)...")
                    
            try:
                chunk_results = await crawler.arun_many(urls=chunk, config=run_config)
                        
                # Count successes and failures
                for result in chunk_results:
                    if result and result.success:
                        results.append(result)
                        successful_crawls += 1
                    else:
                        failed_crawls += 1
                        if result:
                            print(f"   ⚠️  Failed to crawl: {getattr(result, 'url', 'Unknown URL')} - {getattr(result, 'error_message', 'Unknown error')}")
                        
            except Exception as e:
                print(f"   ❌ Error crawling chunk: {str(e)}")
                failed_crawls += len(chunk)
                continue
                    
            time.sleep(0.5)
                
        print(f"📊 Crawl Summary: {successful_crawls} successful, {failed_crawls} failed")

    scraped = []
    processing_errors = 0

    for res in results:
        if not res or not res.success:
            continue

        try:
            page_url = res.url
            raw_md = res.markdown or ""
            
            if not raw_md.strip():
                print(f"   ⚠️  No content found for: {page_url}")
                continue

            # Clean and process text
            cleaned_md = clean_text_bulk(raw_md)
            cleaned_md = clean_markdown_empty_images(cleaned_md)
            cleaned_md = re.sub(r'\s+', ' ', cleaned_md).strip()
            final_text = extract_main_content_from_text(cleaned_md)
            
            # Skip if text is too short after cleaning
            if len(final_text.strip()) < 50:
                print(f"   ⚠️  Content too short for: {page_url}")
                continue

            scraped.append({
                "url": page_url,
                "title": getattr(res, "title", "") or "",
                "text": final_text,
            })

            print(f"✅ {page_url} ({len(final_text)} chars)")
            
        except Exception as e:
            processing_errors += 1
            print(f"   ❌ Error processing {getattr(res, 'url', 'Unknown URL')}: {str(e)}")
            continue
    
    if processing_errors > 0:
        print(f"⚠️  {processing_errors} pages had processing errors")

    # Save results to JSON
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(scraped, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 Scraping completed successfully!")
        print(f"📊 Total pages scraped: {len(scraped)}")
        print(f"💾 Data saved to: {OUTPUT_JSON}")
        
        if len(scraped) == 0:
            print("⚠️  No content was successfully scraped.")
            print("💡 Try checking:")
            print("   - Website accessibility")
            print("   - robots.txt permissions")
            print("   - Website structure/content")
            
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")
        print("💡 The scraping completed but results couldn't be saved to file.")

if __name__ == "__main__":
    asyncio.run(crawl_site())