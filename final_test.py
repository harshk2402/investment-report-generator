"""
This pipeline operates via three-stage process that uses both SEC-API.io's premium services
and Google's Gemini AI. The pipeline begins by using SEC-API.io's Query API to identify the
most recent 10-K filing for each target biotech company (such as MNMD, PTCT, VRTX) by searching
with ticker symbols and form type filters, then retrieving the specific filing URL from the
structured response. Once the filing is located, the system makes individual API calls to SEC-API.io's
Extractor API for each of five critical sections: Item 1 (Business Overview), Item 7
(Management Discussion & Analysis), Item 1A (Risk Factors), Item 2 (Properties), and Item 3 (Legal Proceedings).
Each section is extracted as clean text and combined into a single content block with clear section headers
(=== ITEM X ===) to provide context, resulting in 500k+ characters of targeted, business-relevant content.
This content is then fed into Google's Gemini 1.5 Flash model using a prompt that instructs the AI to act as
a senior biotech analyst, specifically looking for clinical trial keywords like "Phase," "FDA," "endpoint,"
and "investigational" while extracting 24 KPIs including drug names, trial phases, etc. Gemini processes
this structured content and returns detailed JSON arrays containing all discovered trials with complete
metadata, which the pipeline then converts into a standardized pandas DataFrame with proper column ordering
and exports to Excel format
"""

# TO START, JUST PUT IN THE GOOGLE GEMINI API UNDER MAIN FUNCTION!

import pandas as pd
import requests
import json
import time
from typing import Dict, List
import google.generativeai as genai
from datetime import datetime
import re
import adtiam

adtiam.load_creds("adt-sources")


class PremiumSECGeminiExtractor:

    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        # loads sec_api_key here
        self.sec_api_key = adtiam.creds["sources"]["secapid2v"]["key"]

        self.company_data = {
            "MNMD": {"cik": "0001813814", "name": "Mind Medicine (MindMed) Inc"},
            "PTCT": {"cik": "0001070081", "name": "PTC Therapeutics Inc"},
            "BIIB": {"cik": "0000875045", "name": "Biogen Inc"},
            "GILD": {"cik": "0000882095", "name": "Gilead Sciences Inc"},
            "VRTX": {"cik": "0000875320", "name": "Vertex Pharmaceuticals Inc"},
        }

    def get_latest_filing_sections(self, ticker: str) -> tuple:
        company_name = self.company_data.get(ticker, {}).get("name", ticker)

        # Find latest 10-K filing
        query_url = f"https://api.sec-api.io?token={self.sec_api_key}"
        query_payload = {
            "query": f'ticker:{ticker} AND formType:"10-K"',
            "from": 0,
            "size": 1,
            "sort": [{"filedAt": {"order": "desc"}}],
        }

        print(f"  Finding latest 10-K filing for {ticker}...")
        query_response = requests.post(
            query_url, json=query_payload, headers={"Content-Type": "application/json"}
        )
        query_data = query_response.json()

        latest_filing = query_data["filings"][0]
        filing_url = latest_filing["linkToFilingDetails"]

        # Extract sections
        sections_content = ""
        filing_metadata = {
            "accession": latest_filing.get("accessionNo", ""),
            "filing_date": latest_filing.get("filedAt", ""),
            "company_name": company_name,
            "form_type": "10-K",
            "ticker": ticker,
            "cik": self.company_data.get(ticker, {}).get("cik", ""),
        }
        sections_to_extract = [
            "1",
            "7",
            "1A",
            "2",
            "3",
        ]  # the important sections of 10-k report (1,7,MD&A etc)
        extractor_base_url = (
            f"https://api.sec-api.io/extractor?token={self.sec_api_key}"
        )

        for section in sections_to_extract:
            section_payload = {"url": filing_url, "item": section, "type": "text"}

            section_response = requests.get(extractor_base_url, params=section_payload)
            if section_response.status_code == 200:
                section_text = section_response.text
                if section_text and len(section_text.strip()) > 0:
                    print(len(section_text))
                    sections_content += f"\n\n=== ITEM {section} ===\n{section_text}"

            time.sleep(0.5)

        # Clean content
        sections_content = re.sub(r"<[^>]+>", " ", sections_content)
        sections_content = re.sub(r"\s+", " ", sections_content)

        print(f"  Retrieved {len(sections_content)} characters total from all sections")
        # can change this to :500000 to capture more information and more trials (4 more rows, 35 vs 38 for 3 companies I tested)
        return sections_content[:500000], filing_metadata

    def extract_with_advanced_gemini(
        self, ticker: str, content: str, metadata: Dict
    ) -> List[Dict]:
        prompt = f"""
You are a senior biotech analyst extracting comprehensive clinical trial data from {ticker}'s SEC filing.

COMPANY: {ticker} ({metadata.get('company_name', '')})
FILING: {metadata.get('form_type', '')} filed {metadata.get('filing_date', '')}
CIK: {metadata.get('cik', '')}

CONTENT TO ANALYZE:
{content}

TASK: Extract ALL clinical trials and drug development programs with comprehensive details across 24 key performance indicators.

WHAT TO LOOK FOR:
1. Drug names: Look for codes (MM120, VX-548), brand names (Kalydeco, Orkambi), or compound names
2. Indications: Specific diseases being treated (not just "cancer" but "non-small cell lung cancer")
3. Phases: Any mention of Phase 1, 2, 3, 1/2, 2/3, etc.
4. Patient numbers: Any mention of enrollment numbers
5. Study status: ongoing, completed, planned, recruiting, paused, terminated
6. Results: met endpoint, failed endpoint, positive results, etc.
7. Timelines: expected completion dates, data readout dates
8. Study names: specific trial identifiers or names

BIOTECH KEYWORDS TO FOCUS ON:
- "clinical trial", "study", "investigational", "candidate"
- "Phase", "patients", "subjects", "enrolled"
- "endpoint", "efficacy", "safety", "FDA"
- "pivotal", "registrational", "breakthrough", "fast track"
- "topline", "interim", "primary", "secondary"

EXTRACT FORMAT (JSON Array with ALL 24 fields):
[
  {{
    "company": "{ticker}",
    "cik": "{metadata.get('cik', '')}",
    "drug": "specific drug/compound name (e.g., vatiquinone, MM120)",
    "program": "indication/disease being treated",
    "phase": "clinical phase (Phase 1, Phase 2, Phase 3, etc.)",
    "study": "study name or identifier",
    "size": "patient enrollment number only",
    "status_announce": "announced/planned/ongoing/completed",
    "time_period_expected": "timeline or expected completion (e.g., 2023Q2)",
    "explanation": "key details about the trial",
    "primary_endpoint_result": "Met/Not Met/Pending/not specified",
    "adverse_events_summary": "safety findings or AE summary",
    "regulatory_milestone": "FDA meetings, approvals, designations",
    "secondary_endpoint_notes": "secondary endpoint results or notes",
    "trial_design": "study design (randomized, open-label, placebo-controlled, etc.)",
    "biomarkers_used": "biomarkers or companion diagnostics mentioned",
    "comparator_used": "control arm or comparator drug",
    "geography": "study locations (USA, EU, global, etc.)",
    "submission_type": "NDA/BLA/IND/MAA type of regulatory submission",
    "regulatory_track": "fast track, breakthrough, orphan, priority review",
    "milestone_trigger": "what triggered this milestone or next steps",
    "clinical_benefit_summary": "efficacy summary or clinical benefit",
    "readout_type": "topline/interim/final data readout type",
    "trial_status": "current status (recruiting, active, completed, terminated)"
  }}
]

CRITICAL RULES:
- MANDATORY: Every JSON object must contain ALL 24 fields listed above
- Use "not specified" for any field where information is not found in the filing
- Only extract information EXPLICITLY stated in the filing
- Be precise with numbers (patient counts, percentages)
- Create separate entries for each distinct trial
- Include both ongoing and completed studies
- Extract historical trial results if mentioned
- NEVER omit any of the 24 required fields from any trial entry

JSON ARRAY:
"""

        time.sleep(2)
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.05,
                "max_output_tokens": 8000,
            },
        )

        response_text = response.text.strip()
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            trials = json.loads(json_str)

            for trial in trials:
                trial.update(
                    {
                        "accession_number": metadata.get("accession", ""),
                        "filing_date": metadata.get("filing_date", ""),
                        "form_type": metadata.get("form_type", ""),
                        "ticker": metadata.get("ticker", ""),
                        "status_announce": trial.get("status_announce", "extracted"),
                    }
                )

            return trials

        return []

    def run_premium_extraction(self, companies: List[str] = None) -> pd.DataFrame:
        companies = companies or ["MNMD", "PTCT", "VRTX"]
        all_trials = []

        for ticker in companies:
            print(f"\nProcessing {ticker}...")
            content, metadata = self.get_latest_filing_sections(ticker)

            if content:
                print(f"  Extracting clinical trials from {ticker} sections...")
                trials = self.extract_with_advanced_gemini(ticker, content, metadata)
                all_trials.extend(trials)
                print(f"  Extracted {len(trials)} trials for {ticker}")

            time.sleep(3)

        if all_trials:
            column_order = [
                "company",
                "cik",
                "drug",
                "program",
                "phase",
                "study",
                "size",
                "status_announce",
                "time_period_expected",
                "explanation",
                "primary_endpoint_result",
                "adverse_events_summary",
                "regulatory_milestone",
                "secondary_endpoint_notes",
                "trial_design",
                "biomarkers_used",
                "comparator_used",
                "geography",
                "submission_type",
                "regulatory_track",
                "milestone_trigger",
                "clinical_benefit_summary",
                "readout_type",
                "trial_status",
                "accession_number",
            ]

            df = pd.DataFrame(all_trials)
            df = df.reindex(columns=column_order)
            df["size"] = (
                df["size"].astype(str).str.extract(r"(\d+)").fillna("not specified")
            )

            filename = f"clinical_trials_premium_sec_api_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            df.to_excel(filename, index=False)

            print(f"\nEXTRACTION COMPLETE!")
            print(f"Extracted {len(all_trials)} total trials")
            print(f"Saved to: {filename}")

            return df

        return pd.DataFrame()


def main():

    GEMINI_API_KEY = ""

    if not GEMINI_API_KEY:
        print("Please add your Gemini API key")
        return

    extractor = PremiumSECGeminiExtractor(gemini_api_key=GEMINI_API_KEY)
    df = extractor.run_premium_extraction(["MNMD", "PTCT", "VRTX"])

    if not df.empty:
        print("\nSample extracted data:")
        print(df.head())
        print(f"DataFrame shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = main()
