import requests
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin
import json
import time
from tqdm import tqdm
import re

# Use the details page for TOC as it's more reliable for static scraping
TOC_URL = "https://openstax.org/details/books/university-physics-volume-1"
BASE_URL = "https://openstax.org"
OUTPUT_FILE = "data_extraction/openstax_physics_vol1_ch1_6.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ---------------------------
# MATHML TO LATEX CONVERTER
# ---------------------------

def mathml_to_latex(tag):
    """
    Recursively convert a MathML tag (bs4 Tag) to a LaTeX string.
    This is a simplified converter for Presentation MathML found in OpenStax.
    """
    if isinstance(tag, NavigableString):
        return tag.strip()
    
    if tag.name == 'math':
        # Process children
        content = "".join([mathml_to_latex(c) for c in tag.children])
        return f"${content}$"
    
    if tag.name == 'mrow':
        return "".join([mathml_to_latex(c) for c in tag.children])
    
    if tag.name == 'mi': # Identifier
        text = tag.get_text(strip=True)
        # Handle greek letters or special symbols if needed, but usually text is fine
        return text
    
    if tag.name == 'mn': # Number
        return tag.get_text(strip=True)
    
    if tag.name == 'mo': # Operator
        op = tag.get_text(strip=True)
        # Map common operators to latex if needed
        return op
    
    if tag.name == 'mfrac':
        children = [c for c in tag.children if c.name]
        if len(children) >= 2:
            num = mathml_to_latex(children[0])
            den = mathml_to_latex(children[1])
            return f"\\frac{{{num}}}{{{den}}}"
        return ""

    if tag.name == 'msup':
        children = [c for c in tag.children if c.name]
        if len(children) >= 2:
            base = mathml_to_latex(children[0])
            sup = mathml_to_latex(children[1])
            return f"{{{base}}}^{{{sup}}}"
        return ""
    
    if tag.name == 'msub':
        children = [c for c in tag.children if c.name]
        if len(children) >= 2:
            base = mathml_to_latex(children[0])
            sub = mathml_to_latex(children[1])
            return f"{{{base}}}_{{{sub}}}"
        return ""
    
    if tag.name == 'msubsup':
        children = [c for c in tag.children if c.name]
        if len(children) >= 3:
            base = mathml_to_latex(children[0])
            sub = mathml_to_latex(children[1])
            sup = mathml_to_latex(children[2])
            return f"{{{base}}}_{{{sub}}}^{{{sup}}}"
        return ""

    if tag.name == 'msqrt':
        content = "".join([mathml_to_latex(c) for c in tag.children])
        return f"\\sqrt{{{content}}}"

    if tag.name == 'mtext':
        return f"\\text{{{tag.get_text(strip=True)}}}"
    
    # Fallback: just return text of children
    return "".join([mathml_to_latex(c) for c in tag.children])

# ---------------------------
# CRAWLER LOGIC
# ---------------------------

def get_section_urls():
    import json
    import os
    
    json_path = "data_extraction/toc_urls.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please populate it first.")
        return []

    print(f"Loading TOC from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        section_urls = json.load(f)
    
    print(f"Loaded {len(section_urls)} section URLs locally")
    return section_urls

def filter_chapters_1_to_6(urls):
    filtered = []
    for url in urls:
        # Expected format: .../pages/1-introduction or .../pages/1-2-units ...
        try:
            slug = url.split("/pages/")[1]
            # Chapter number is the first part before the dash
            # Handle cases like "1-introduction" vs "source-introduction" (if any)
            if slug[0].isdigit():
                chapter = int(slug.split('-')[0])
                if 1 <= chapter <= 6:
                    filtered.append(url)
        except:
            continue
    return sorted(filtered)

def extract_content(url):
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        
        main = soup.find("main") or soup.find("div", {"data-type": "chapter"}) or soup.find("div", {"data-type": "page"})
        if not main:
            return None

        # Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Unknown"

        # 1. Convert MathML to LaTeX
        # Find all <math> tags
        for math_tag in main.find_all("math"):
            try:
                latex = mathml_to_latex(math_tag)
                # Replace the math tag with the latex string
                # We wrap it in spaces to avoid stickiness
                math_tag.replace_with(f" {latex} ")
            except Exception as e:
                pass # print(f"Math conversion error: {e}")

        # 2. Extract text blocks
        blocks = []
        # We want paragraphs, headings, lists, problem boxes
        # OpenStax structure: <p>, <ul>, <div class="problem">, etc.
        # Simplest approach: iterate over relevant tags in order
        
        target_tags = ["p", "h2", "h3", "h4", "li", "div.problem", "div.example"]
        
        # A more robust linear extraction:
        # We can just get_text() but that loses structure (e.g. merging headers with text).
        # Let's iterate over children of main recursively or use a generator.
        
        # Simplified: Get all text from main with strict separator
        text = main.get_text(separator="\n\n", strip=True)
        
        # Post-processing to clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        page_slug = url.split("/pages/")[1]
        chapter = int(page_slug.split('-')[0]) if page_slug[0].isdigit() else 0

        return {
            "url": url,
            "title": title,
            "chapter": chapter,
            "content": text
        }

    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return None

def main():
    urls = get_section_urls()
    urls = filter_chapters_1_to_6(urls)
    print(f"Targeting {len(urls)} sections from Chapter 1 to 6.")
    
    data = []
    for url in tqdm(urls):
        page_data = extract_content(url)
        if page_data:
            data.append(page_data)
        time.sleep(0.2)
        
    print(f"Extracted {len(data)} pages.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
