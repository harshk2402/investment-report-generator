import os
import json
import time
from datetime import datetime
from typing import List, Dict

import pandas as pd
import google.generativeai as genai
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from google.generativeai.types import content_types

# ✅ Set up Gemini API
GEMINI_API_KEY = os.getenv("google_api_key2") or "you_api_key"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-pro")

# Ignore warnings
os.environ["GRPC_VERBOSITY"] = "NONE"


# ✅ Gemini: Determine if the title is a Press Release
def is_title_press_release(title: str) -> str:
    prompt = f"""
You are a financial assistant helping filter company-issued press releases from other types of news.

Given a single **news article title**, determine whether it is a **formal press release issued by the company** (such as earnings reports, product announcements, FDA updates, clinical trial results, corporate updates, investor event participation, etc.)

Do NOT classify third-party reports, law firm investigations, market research, or analyst commentary as press releases.

Respond ONLY with one of the following:
- "Press Release"
- "Not Press Release"

Title: "{title}"
Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else "None"


def scrape_press_release(
    ticker: str, chromedriver_path: str, save_dir: str, start_date: str = "2023-01-01"
) -> List[Dict]:
    print(f"\n🚀 Start scraping press releases for {ticker} (since {start_date})...")
    options = Options()
    options.add_argument("--headless")  # Run in headless mode (no GUI)
    options.add_argument("--disable-gpu")  # Optional: improves compatibility
    options.add_argument(
        "--window-size=1920,1080"
    )  # Optional: ensures full layout loads

    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    base_url_template = f"https://www.globenewswire.com/en/search/keyword/{ticker}/load/before?page={{page}}&pageSize=100"

    press_list = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    page = 1

    while True:
        url = base_url_template.format(page=page)
        print(f"\n📄 Scraping page {page}: {url}")
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "mainLink"))
        )
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.find_all("div", class_="mainLink")
        if not articles:
            print("⚠️ No articles found, stopping.")
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
                    print(
                        f"🛑 {pub_date_str} is earlier than {start_date}, stopping current and following pages."
                    )
                    stop_scraping = True
                    break

                print(f"\n📰 Article {idx+1}: {title}")
                print(f"🗓️ Publish Date: {pub_date_str}")

                label = is_title_press_release(title)
                print(f"🤖 Gemini Result: {label}")
                if label != "Press Release":
                    print("⛔ Filtered out, skipping.")
                    continue

                # ➤ Get full article text
                driver.get(link)
                time.sleep(2)
                article_html = driver.page_source
                article_soup = BeautifulSoup(article_html, "html.parser")
                content_div = article_soup.find("div", class_="main-container-content")
                full_text = (
                    content_div.get_text(separator="\n", strip=True)
                    if content_div
                    else "N/A"
                )

                press_list.append(
                    {
                        "ticker": ticker,
                        "title": title,
                        "link": link,
                        "date": pub_date_str,
                        "text": full_text,
                    }
                )

                # Go back to search result page
                driver.get(url)
                time.sleep(1)

            except Exception as e:
                print(f"❌ Error occurred: {e}")
                continue

        if stop_scraping:
            break

        page += 1

    driver.quit()

    # ✅ Save the result for this company
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, f"{ticker.lower()}_press_releases.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(press_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Done with {ticker}, saved {len(press_list)} press releases.")
    return press_list


if __name__ == "__main__":
    chromedriver_path = "/opt/homebrew/bin/chromedriver"
    save_dir = "press_data"
    tickers = [
        "PTCT",
        "MNMD",
        "VTGN",
        "OVID",
        "PRAX",
        "CNTA",
        "ATAI",
        "TARS",
        "PTGX",
        "MDGL",
    ]  # take out RNA

    all_data = []

    for ticker in tickers:
        data = scrape_press_release(
            ticker, chromedriver_path, save_dir, start_date="2025-01-01"
        )
        all_data.extend(data)

    all_data

    print(f"\n🎉 All done, total {len(all_data)} press releases scraped.")
