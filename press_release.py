import os
import time
from datetime import datetime
from typing import List, Dict

import google.generativeai as genai
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import faiss_manager

# Configure Gemini API
GEMINI_API_KEY = os.getenv("google_api_key2") or "your_api_key_here"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-pro")

# Suppress warnings
os.environ["GRPC_VERBOSITY"] = "NONE"


def classify_press_release_titles(titles: List[str]) -> List[str]:
    """
    Classify a list of article titles, returning only those that
    are press releases, according to the LLM.

    The LLM returns a numbered list of titles that are press releases.
    """
    if not titles:
        return []

    prompt = (
        "You are a financial assistant helping filter company-issued press releases from other types of news.\n\n"
        "Given the following list of news article titles, identify which are formal press releases issued by the company (e.g., earnings reports, product announcements, FDA updates, clinical trial results, corporate updates, investor event participation).\n"
        "Do NOT consider third-party reports, law firm investigations, market research, or analyst commentary as press releases.\n\n"
        "Respond ONLY with a numbered list of the titles that ARE press releases, preserving their exact wording and order.\n\n"
        "Example response format:\n\n"
        "1. Earnings Report Q2 2025\n"
        "2. ABC Corp Announces FDA Approval\n"
        "3. New Product Launch by XYZ Corp\n\n"
        "Titles:\n"
    )
    for i, title in enumerate(titles):
        prompt += f"{i+1}. {title}\n"

    response = model.generate_content(prompt)

    if not response.candidates or not response.candidates[0].content.parts:
        print("Warning: no response from Gemini, skipping.")
        return []

    text = response.text.strip()

    # Parse response lines that start with a number and extract the title text
    press_release_titles = []
    for line in text.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            # Remove number and dot prefix
            title_text = line.split(".", 1)[1].strip()
            press_release_titles.append(title_text)

    return press_release_titles


def scrape_press_release(
    ticker: str, chromedriver_path: str, start_date: str = "2023-01-01"
) -> tuple[list, list]:
    print(f"\nScraping press releases for {ticker} (since {start_date})...")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    base_url_template = f"https://www.globenewswire.com/en/search/keyword/{ticker}/load/before?page={{page}}&pageSize=100"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    page = 1

    all_articles = []  # Store all titles + links + dates

    while True:
        url = base_url_template.format(page=page)
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "mainLink"))
        )
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.find_all("div", class_="mainLink")
        if not articles:
            break

        stop_scraping = False

        for idx, a in enumerate(articles):
            try:
                anchor = a.find("a")
                title = anchor.text.strip()
                relative_link = anchor["href"]
                link = "https://www.globenewswire.com" + relative_link

                # Get publish date
                parent = a.find_parent("li")
                date_span = parent.find("div", class_="date-source").find("span")
                pub_date_str = date_span.text.strip() if date_span else "N/A"
                pub_date = datetime.strptime(
                    pub_date_str.split(" ET")[0], "%B %d, %Y %H:%M"
                )

                if pub_date < start_dt:
                    stop_scraping = True
                    break

                # Collect article metadata only, no article text yet
                all_articles.append(
                    {
                        "ticker": ticker,
                        "title": title,
                        "link": link,
                        "filing_date": pub_date_str,
                    }
                )

            except Exception as e:
                print(f"❌ Error occurred: {e}")
                continue

        if stop_scraping:
            break

        page += 1

    print(f"Sending {len(all_articles)} titles to Gemini for batch classification...")

    titles = [entry["title"] for entry in all_articles]
    classified_press_releases = classify_press_release_titles(titles)

    # Filter to keep only those metadata entries matching classified press release titles
    press_releases_metadata = [
        entry for entry in all_articles if entry["title"] in classified_press_releases
    ]

    # Filter out those metadata entries that are already indexed
    indexed_hash_tracker = faiss_manager.FilingHashTracker()
    press_releases_metadata = [
        entry
        for entry in press_releases_metadata
        if not indexed_hash_tracker.is_indexed(
            entry["ticker"],
            "press release",
            entry["filing_date"],
        )
    ]

    press_list = []
    metadata_list = []

    # Fetch full article text only for press releases
    print(f"Fetching full text for {len(press_releases_metadata)} press releases...")
    for entry in press_releases_metadata:
        try:
            driver.get(entry["link"])
            time.sleep(2)
            article_html = driver.page_source
            article_soup = BeautifulSoup(article_html, "html.parser")
            content_div = article_soup.find("div", class_="main-container-content")
            full_text = (
                content_div.get_text(separator="\n", strip=True)
                if content_div
                else "N/A"
            )
            press_list.append(full_text)
            metadata_list.append(entry)
        except Exception as e:
            print(f"❌ Error fetching article text for {entry['link']}: {e}")
            continue

    driver.quit()

    print(f"✅ Done with {ticker}, found {len(press_list)} press releases.")
    return press_list, metadata_list


def get_press_releases(tickers: List[str], start_date: str) -> tuple[list, list]:
    chromedriver_path = "/opt/homebrew/bin/chromedriver"

    all_press_data = []
    all_metadata = []

    for ticker in tickers:
        press_data, metadata = scrape_press_release(
            ticker, chromedriver_path, start_date=start_date
        )
        all_press_data.extend(press_data)
        all_metadata.extend(metadata)

    return all_press_data, all_metadata
