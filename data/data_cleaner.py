import json
import re
from collections import Counter
from typing import List, Dict, Set
import difflib

class DataCleaner:
    def __init__(self):
        self.repetitive_patterns = [
            # Navigation/menu patterns
            r'\* Services \*',
            r'\* À Propos \*',
            r'\* Contactez-Nous',
            r'Skip to content',
            r'du Site \* \* Services',
            
            # Footer/copyright patterns
            r'Copyright © 2025 - DigiSand',
            r'Phone: \+212 695 96 66 63',
            r'Fax: \+212 528 87 00 41',
            r'contact@digisand\.ma',
            r'Besoin d\'aide ou avez-vous une question',
            r'Statcounter',
            
            # Company description repetition
            r'Digisand est une société spécialisée dans le développement informatique',
            r'le digital signage et une gamme étendue de services numériques',
            r'Nous offrons des solutions sur mesure pour aider les entreprises',
            r'à exploiter pleinement le potentiel de la technologie numérique',
            
            # Common phrases
            r'Contactez-Nous Phone:',
            r'Besoin d\'une question \?',
            r'À Propos \* Contactez-Nous'
        ]
    
    def load_data(self, filepath: str) -> List[Dict]:
        """Load scraped data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return []
    
    def save_data(self, data: List[Dict], filepath: str):
        """Save cleaned data to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Data saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def remove_repetitive_patterns(self, text: str) -> str:
        """Remove common repetitive patterns from text."""
        cleaned_text = text
        
        # Remove repetitive patterns
        for pattern in self.repetitive_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
    
    def remove_duplicate_pages(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicate pages based on URL normalization."""
        seen_urls = set()
        unique_pages = []
        
        for page in data:
            url = page.get('url', '')
            # Normalize URL by removing fragments and query params for comparison
            normalized_url = re.sub(r'[#\?].*$', '', url)
            
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_pages.append(page)
        
        print(f"🗑️  Removed {len(data) - len(unique_pages)} duplicate pages")
        return unique_pages
    
    def find_similar_content(self, texts: List[str], threshold: float = 0.8) -> List[int]:
        """Find indices of pages with highly similar content."""
        similar_indices = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
                if similarity > threshold:
                    similar_indices.extend([i, j])
        
        return list(set(similar_indices))
    
    def remove_content_duplicates(self, data: List[Dict]) -> List[Dict]:
        """Remove pages with nearly identical content."""
        if len(data) <= 1:
            return data
        
        # Get texts for comparison
        texts = [page.get('text', '') for page in data]
        
        # Find similar content
        similar_indices = self.find_similar_content(texts, threshold=0.85)
        
        # Keep only unique content (keep first occurrence)
        seen_content_hashes = set()
        unique_content_pages = []
        
        for i, page in enumerate(data):
            content_hash = hash(page.get('text', '')[:200])  # Hash first 200 chars
            
            if i not in similar_indices or content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique_content_pages.append(page)
        
        print(f"🗑️  Removed {len(data) - len(unique_content_pages)} pages with duplicate content")
        return unique_content_pages
    
    def clean_text_content(self, data: List[Dict]) -> List[Dict]:
        """Clean text content of each page."""
        cleaned_data = []
        
        for page in data:
            cleaned_page = page.copy()
            
            # Clean the text
            original_text = page.get('text', '')
            cleaned_text = self.remove_repetitive_patterns(original_text)
            
            # Remove very short content
            if len(cleaned_text.strip()) < 50:
                print(f"⚠️  Skipping page with very short content: {page.get('url', 'Unknown')}")
                continue
            
            cleaned_page['text'] = cleaned_text
            cleaned_page['original_length'] = len(original_text)
            cleaned_page['cleaned_length'] = len(cleaned_text)
            cleaned_page['reduction_percentage'] = round(
                (1 - len(cleaned_text) / len(original_text)) * 100, 2
            ) if original_text else 0
            
            cleaned_data.append(cleaned_page)
        
        return cleaned_data
    
    def generate_statistics(self, original_data: List[Dict], cleaned_data: List[Dict]):
        """Generate cleaning statistics."""
        print("\n📊 CLEANING STATISTICS:")
        print("=" * 50)
        print(f"Original pages: {len(original_data)}")
        print(f"Cleaned pages: {len(cleaned_data)}")
        print(f"Pages removed: {len(original_data) - len(cleaned_data)}")
        
        if original_data and cleaned_data:
            total_original_chars = sum(len(page.get('text', '')) for page in original_data)
            total_cleaned_chars = sum(len(page.get('text', '')) for page in cleaned_data)
            
            print(f"\nText reduction:")
            print(f"  Original characters: {total_original_chars:,}")
            print(f"  Cleaned characters: {total_cleaned_chars:,}")
            if total_original_chars > 0:
                reduction = (1 - total_cleaned_chars / total_original_chars) * 100
                print(f"  Reduction: {reduction:.1f}%")
        
        # Show reduction per page
        print(f"\nPer-page reduction:")
        for page in cleaned_data:
            reduction = page.get('reduction_percentage', 0)
            url = page.get('url', 'Unknown')
            short_url = url[:50] + ('...' if len(url) > 50 else '')
            print(f"  {short_url}: {reduction}% reduction"),
    
    def clean_dataset(self, input_file: str = "scraped_data.json", 
                     output_file: str = "cleaned_data.json"):
        """Main cleaning function."""
        print("🧹 Starting data cleaning process...")
        print("=" * 50)
        
        # Load data
        data = self.load_data(input_file)
        if not data:
            print("❌ No data to clean. Exiting.")
            return
        
        print(f"📄 Loaded {len(data)} pages")
        
        # Step 1: Remove duplicate URLs
        data = self.remove_duplicate_pages(data)
        
        # Step 2: Remove content duplicates
        data = self.remove_content_duplicates(data)
        
        # Step 3: Clean text content
        cleaned_data = self.clean_text_content(data)
        
        # Step 4: Generate statistics
        self.generate_statistics(data, cleaned_data)
        
        # Step 5: Save cleaned data
        self.save_data(cleaned_data, output_file)
        
        print(f"\n🎉 Data cleaning completed!")
        print(f"📁 Output file: {output_file}")

def main():
    """Main function to run the data cleaner."""
    cleaner = DataCleaner()
    
    # Clean the default scraped data
    cleaner.clean_dataset()
    
    # Optionally, clean a specific file
    # cleaner.clean_dataset("my_scraped_data.json", "my_cleaned_data.json")

if __name__ == "__main__":
    main()