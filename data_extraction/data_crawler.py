import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
from tqdm import tqdm

BASE_BOOK_URL = "https://openstax.org/books/university-physics-volume-1"
BOOK_SLUG = "/books/university-physics-volume-1/pages/"
OUTPUT_FILE = "openstax_physics_vol1_ch1_6.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Academic Research Bot)"
}

# ---------------------------
# STEP 1: Get all section URLs from TOC
# ---------------------------

def get_section_urls():
    print("Fetching TOC...")
    r = requests.get(BASE_BOOK_URL, headers=HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    links = soup.find_all("a", href=True)

    section_urls = set()

    for a in links:
        href = a["href"]
        if BOOK_SLUG in href:
            full_url = urljoin("https://openstax.org", href)
            section_urls.add(full_url)

    print(f"Found {len(section_urls)} section URLs")
    return sorted(section_urls)

# ---------------------------
# STEP 2: Filter Chapters 1–6
# ---------------------------

def is_chapter_1_to_6(url):
    """
    Example URLs:
    /pages/1-introduction
    /pages/2-1-displacement
    /pages/6-4-rotational-work
    """
    try:
        page_part = url.split("/pages/")[1]
        chapter_num = int(page_part.split("-")[0])
        return 1 <= chapter_num <= 6
    except:
        return False

# ---------------------------
# STEP 3: Extract content from a section page
# ---------------------------

def extract_section_content(url):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    main = soup.find("main")
    if not main:
        return None

    # Title
    title_tag = main.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    # Extract paragraphs only
    paragraphs = []
    for p in main.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) > 30:  # remove very short junk
            paragraphs.append(text)

    if not paragraphs:
        return None

    # Infer chapter number
    try:
        chapter_num = int(title.split(".")[0])
    except:
        chapter_num = None

    return {
        "url": url,
        "chapter": chapter_num,
        "section_title": title,
        "text": "\n".join(paragraphs),
        "source": "OpenStax University Physics Volume 1",
        "license": "CC BY 4.0",
        "attribution": "Access for free at openstax.org."
    }

# ---------------------------
# STEP 4: Main crawl loop
# ---------------------------

def crawl():
    urls = get_section_urls()
    urls = [u for u in urls if is_chapter_1_to_6(u)]

    print(f"Crawling {len(urls)} sections (Ch 1–6)...")

    data = []

    for url in tqdm(urls):
        try:
            section = extract_section_content(url)
            if section:
                data.append(section)
            time.sleep(0.5)  # polite crawling
        except Exception as e:
            print(f"Failed on {url}: {e}")

    print(f"Extracted {len(data)} sections")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_FILE}")

# ---------------------------
# RUN
# ---------------------------

if __name__ == "__main__":
    crawl()
