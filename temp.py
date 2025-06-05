import requests
from bs4 import BeautifulSoup

urls = [
    "https://www.sec.gov/Archives/edgar/data/1689548/000168954825000058/prax-20250331.htm",
    "https://www.sec.gov/Archives/edgar/data/1689548/000168954825000040/prax-20241231.htm",
    "https://www.sec.gov/Archives/edgar/data/1689548/000168954824000101/prax-20240930.htm",
    "https://www.sec.gov/Archives/edgar/data/1689548/000168954824000088/prax-20240630.htm",
]
headers = {"User-Agent": "BiotechCatalystBot/1.0 (your.email@domain.com)"}
for url in urls:
    response_html = requests.get(url, headers=headers)
    if response_html.status_code != 200:
        print(f"Failed to retrieve {url}. Status: {response_html.status_code}")
        continue

    soup = BeautifulSoup(response_html.text, "html.parser")
    article_text = soup.get_text(separator="\n")
