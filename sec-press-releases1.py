from bs4 import BeautifulSoup

# Path to your HTML file
html_file = "data/prax-20231128.html"

with open(html_file, "r", encoding="cp1252") as f:
    soup = BeautifulSoup(f, "html.parser")

# Find all a elements with style attribute containing "-sec-extract:exhibit"
exhibit_links = soup.select("a[style*='-sec-extract:exhibit']")

# Create a list of dictionaries containing text and first link
result_list = []
for link in exhibit_links:
    # Get the text content
    text = link.get_text(strip=True)
    
    # Get the href attribute (first link)
    href = link.get("href", "")
    
    # Create dictionary and add to list
    link_dict = {
        "text": text,
        "link": href
    }
    result_list.append(link_dict)

# Consolidate the list by pairing elements
consolidated_list = [
    {
        'exhibit': result_list[i]['text'],
        'description': result_list[i + 1]['text'],
        'url': result_list[i]['link']
    }
    for i in range(0, len(result_list), 2)
    if i + 1 < len(result_list)
]

# Print consolidated results
print(f"\nConsolidated {len(consolidated_list)} exhibit pairs:")
for item in consolidated_list:
    print(f"Exhibit: {item['exhibit']}")
    print(f"Description: {item['description']}")
    print(f"URL: {item['url']}")
    print("-" * 50)

