import requests
from bs4 import BeautifulSoup

URL = "https://openstax.org/details/books/university-physics-volume-1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

r = requests.get(URL, headers=HEADERS)
print(f"Status: {r.status_code}")
print(f"Length: {len(r.text)}")

soup = BeautifulSoup(r.text, "lxml")
links = soup.select('.table-of-contents a')
print(f"Found {len(links)} links in .table-of-contents")
if len(links) > 0:
    print("Sample link:", links[0]['href'])

# Print all links containing 'pages'
all_links = soup.find_all('a', href=True)
pages_links = [a['href'] for a in all_links if 'pages' in a['href']]
print(f"Found {len(pages_links)} links with 'pages'")
if len(pages_links) > 0:
    print("Sample page link:", pages_links[0])
