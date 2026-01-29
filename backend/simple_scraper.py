import asyncio
import json
import requests
from bs4 import BeautifulSoup
from typing import Dict
import re

def scrape_single_page(url: str) -> Dict:
    """Scrape a single page using requests and beautifulsoup4."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Get text content
        text = soup.get_text()
        
        # Clean the text
        cleaned_text = clean_scraped_text(text)
        
        return {
            "url": url,
            "title": title_text,
            "text": cleaned_text,
            "success": True
        }
        
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "success": False
        }

def clean_scraped_text(text: str) -> str:
    """Clean scraped text by removing boilerplate content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common menu/footer words
    menu_words = [
        r"\bHome\b", r"\bAbout\b", r"\bContact\b", r"\bPrivacy\b",
        r"\bTerms\b", r"\bLogin\b", r"\bSign Up\b", r"\bRegister\b",
        r"\bCopyright\b", r"\b©\b", r"\bAll rights reserved\b",
        r"\bNavigation\b", r"\bMenu\b", r"\bFooter\b"
    ]
    text = re.sub("|".join(menu_words), " ", text, flags=re.IGNORECASE)
    
    # Clean up spacing again
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_website_to_json(url: str, output_file: str = "scraped_data.json"):
    """Scrape a website and save results to JSON file."""
    print(f"🕷️  Scraping: {url}")
    
    # Scrape the main page
    result = scrape_single_page(url)
    
    if result["success"]:
        # Save to JSON
        data = [result]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved scraped data to {output_file}")
        return True
    else:
        print(f"❌ Failed to scrape: {result['error']}")
        return False

# Example usage
if __name__ == "__main__":
    url = input("Enter website URL to scrape: ").strip()
    if url:
        scrape_website_to_json(url)