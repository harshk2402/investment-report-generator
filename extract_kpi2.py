import pandas as pd
from schema import EventList, ValidationFeedback
import adtiam
import os
from typing import List
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import common
import math
import time

load_dotenv()
adtiam.load_creds("adt-llm")
os.environ["OPENAI_API_KEY"] = adtiam.creds["llm"]["openai"]


def get_validation_prompt(
    original_text_chunks: List[str],
    extracted_data: EventList,
) -> str:
    today = datetime.now()
    formatted_date = today.strftime("%Y-%m-%d")

    validation_prompt = f"""
You are an expert biomedical research analyst with a meticulous eye for detail.
Your task is to **validate** the provided `extracted_data` against the `original_source_text`.

Strictly follow these rules for validation:
- **Accuracy Check:** Verify every piece of information in the `extracted_data` against the `original_source_text` based on the original extraction rules:
  - **Hallucination Check:** Is all extracted information **explicitly stated, unambiguously implied, or correctly and reasonably inferable** from the `original_source_text`?  Report if any data is not clearly supported by the text.
  - **Completeness Check:** Is all relevant information from the `original_source_text` that **should have been extracted according to the initial extraction schema's rules** present in the `extracted_data`? Report any missing relevant data.
  - **Value Accuracy Check:** Are all extracted values precisely as stated or accurately derived from the `original_source_text` (e.g., correct numbers, dates, statuses)? Report any inaccuracies.
- **Schema Adherence:** Ensure the `extracted_data` fully conforms to the `EventList` and `EventCatalyst` schema definitions (e.g., correct field names, data types, use of 'not specified' where appropriate).
- **Correction:** If any inaccuracies, missing data, or hallucinations are found, provide a `corrected_data` field in your output.
  - The `corrected_data` field should contain the *entire*, revised `EventList` object with all necessary corrections applied, strictly adhering to the original extraction rules.
  - If no corrections are needed (i.e., `is_accurate` is True), this field should be `null`.
- **Feedback Structure:** Output a JSON object conforming exactly to the `ValidationFeedback` schema below.
  - IMPORTANT: If `is_accurate` is False, the `issues_found` list must contain detailed descriptions of each issue.
  - For each issue, specify its `type` (e.g., 'missing_data', 'hallucination', 'inaccurate_value', 'schema_violation') and a clear `description` including suggested corrections.
  - If `is_accurate` is True, `issues_found` must be an empty list `[]`.

Current Date: {formatted_date}

[Original Source Text]
{" ".join(original_text_chunks)}

[Extracted Data (JSON)]
{json.dumps(extracted_data.model_dump(), indent=2)}

[ValidationFeedback Schema]
{ValidationFeedback.model_json_schema()}

Output the result as a JSON object of `ValidationFeedback`.
"""

    print(len(validation_prompt), " - characters in validation prompt")

    return validation_prompt


def batched_validate_output(search_chunks: List[str], result: EventList, batch_size=5):

    df_metrics = pd.DataFrame()
    os.environ["GOOGLE_API_KEY"] = os.getenv("google_api_key3") or "your_api_key_here"

    client = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    structured_client = client.with_structured_output(ValidationFeedback)

    validated_events = []
    seen_keys = set()
    all_events = result.events
    total_batches = math.ceil(len(all_events) / batch_size)

    for i in range(total_batches):
        batch_events = all_events[i * batch_size : (i + 1) * batch_size]
        partial_result = EventList(events=batch_events)
        validation_prompt = get_validation_prompt(search_chunks, partial_result)

        try:
            validation_response = structured_client.invoke(validation_prompt)

            if validation_response:
                validation_result = ValidationFeedback.model_validate(
                    validation_response
                )
                # Choose corrected data if inaccurate, else use original batch
                events_to_add = (
                    EventList.model_validate(validation_result.corrected_data).events
                    if not validation_result.is_accurate
                    and validation_result.corrected_data
                    else batch_events
                )

                # Deduplicate based on identity key
                for event in events_to_add:
                    key = common.event_identity_key(event)
                    if key not in seen_keys:
                        validated_events.append(event)
                        seen_keys.add(key)
        except Exception as e:
            print(f"Validation error in batch {i+1}: {e}")
            validated_events.extend(batch_events)  # fallback: accept originals

        time.sleep(0.5)

    df_metrics = pd.DataFrame([e.model_dump() for e in validated_events])
    return df_metrics


def extract_kpi(
    search_metric, search_chunks, writer_raw: pd.ExcelWriter, company: str = "Unknown"
):
    today = datetime.now()
    formatted_date = today.strftime("%Y-%m-%d")

    # Construct LLM prompt
    llm_prompt = f"""
You are a biomedical research analyst. Your task is to extract and structure **all clinically and regulatorily relevant**
drug development and trial data from the following biotech disclosure text (e.g., financial filings, press releases).

You must output a **JSON list of objects** that conforms exactly to the schema below. Your output must adhere to the following strict rules:
- Include **all** fields in the schema — never omit or ignore any field.
- Use the **exact field names** as defined — spelling and casing must match.
- Populate each field **only** if the information is explicitly stated or unambiguously implied in the text.
- If a field's information is not present, return the string `'not specified'` (not `null`, empty, or missing).
- Do **not infer**, **assume**, or **hallucinate** any data. Only report what is clearly given.
- Do **not duplicate** entries or fields across the output.
- EXTREMELY IMPORTANT: Return one object per distinct drug development program or major clinical trial that is mentioned in the text, regardless of whether it directly mentions the words in {search_metric}. Do not restrict output based on the search_metric — use it as a loose topical guide, not a filter.** For each such relevant program/trial, a "distinct" object is warranted for each:
    - **Unique drug candidate** (a different molecule, biologic, or therapeutic modality).
    - **New disease indication** being pursued for an existing drug candidate.
    - **Major, separately-named clinical trial** (e.g., a pivotal proof-of-concept, Phase 2b, or Phase 3 study) even for an existing drug-indication pair, as these represent discrete, high-impact data readouts and potential value inflection points.
- Be concise and consistent, but thorough — extract everything real, nothing imagined.
- Treat strings as case-insensitive when comparing or grouping values (e.g., "Phase 2a" and "phase 2A" are the same).

[General Instructions]
**ABSOLUTE PRIORITY RULE: For any given clinical trial or event that is relevant to the search metric, ALWAYS extract and return the LATEST, MOST CURRENT, and DEFINITIVELY CONFIRMED information. If a result or status was once 'expected' but the text later states it 'was announced' or 'demonstrated' a specific outcome, you MUST use the announced/demonstrated status and result.**
Crucial Rule: ONLY use 'not specified' for a field if the information is genuinely ABSENT from the provided text for that specific event. Do NOT use 'not specified' if the information can be reasonably inferred or is directly stated elsewhere in the text for that entry
Current Date: {formatted_date}

[schema]
EventCatalyst:
company: The exact, full name of the company from text (e.g., 'Taysha Gene Therapies, Inc.'). Do not confuse this field with a ticker symbol.
accession_number: The SEC Accession Number that uniquely identifies a specific filing submission, e.g., 0001689548-23-000044 
drug: Name of the drug, e.g., ulixacaltamide  
program: The indication or program, e.g., ENERGY Program  
phase: Clinical phase only (exclude study name), e.g., Phase 2a  
study: Study identifier or title only (exclude phase), e.g., Photo-Paroxysmal Response study  
size: Number of enrolled patients, e.g., 1000  
status_announce: Use 'Announced' (Actual/Announced) ONLY if the results/event have been definitively announced (e.g., 'results were announced', 'demonstrated positive results'). Use 'Expected' if the event is still anticipated in the future relative to 'Current Date' (e.g., 'expected in Q3 2025'). **If an event was 'expected' at a date BEFORE 'Current Date' AND no 'Actual/Announced' result is found in the text, maintain 'Expected' as the status (indicating an overdue expectation).** Use "Planned" if the trial is described as being planned or will be initiated. Default to 'not specified' if no clear status is mentioned.
time_period_expected: Time results were announced or are expected, e.g., 2025Q2, 2025H2  
explanation: Summary or context on the trial results or disclosure  
primary_endpoint_result: Extract the direct outcome of the primary endpoint. Use 'Positive', 'Not Met', 'Futility' (if explicitly stated), or 'not specified'. Do NOT use 'not specified' if a result is implied (e.g., 'demonstrated positive results'). 
adverse_events_summary: Summary of observed AEs or safety issues, e.g., Mild TEAEs, No SAEs  
regulatory_milestone: Most recent regulatory development, e.g., NDA Submitted  
secondary_endpoint_notes: Any noted results from secondary endpoints  
trial_design: Trial structure, e.g., randomized, double-blind, placebo-controlled  
biomarkers_used: Biomarkers included for monitoring or stratification  
comparator_used: Control arm, e.g., Placebo, Standard of Care  
geography: Countries or regions involved in trial or submission  
submission_type: Regulatory submission type, e.g., NDA, MAA, BLA  
regulatory_track: Review designation, e.g., Priority Review, Accelerated Approval  
milestone_trigger: Any financial/partnership milestone triggered, e.g., Novartis $1B upfront payment  
clinical_benefit_summary: Summary of clinical benefit, e.g., Dose-dependent HTT lowering  
readout_type: Type of result readout, e.g., Topline, Interim, Final  
trial_status: Use 'Ongoing' if the trial is actively recruiting, dosing, monitoring, or if the company has explicitly stated they are 'continuing' the study. Use 'Completed' ONLY if the trial has definitively finished all activities, including data collection and analysis, for that specific phase. Use 'Enrolling' if specifically stated. Use 'Planned' if specifically stated. Default to 'not specified' if no clear status is mentioned.

EventList:
events: List of EventCatalyst objects

[data]
{" ".join(search_chunks)}

Output the result as a JSON list of `EventCatalyst` objects, like:
[
  EventCatalyst(...),
  EventCatalyst(...)
]
"""
    print()
    print(len(llm_prompt), " - characters in prompt")
    print("Using Gemini API to extract KPIs...")

    os.environ["GOOGLE_API_KEY"] = os.getenv("google_api_key") or "your_api_key_here"
    client = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    structured_client = client.with_structured_output(EventList)

    response = structured_client.invoke(llm_prompt)

    result = None
    if response is None:
        print("No response from Gemini, retrying in 5s")
        time.sleep(5)
        response = structured_client.invoke(llm_prompt)

    if response is None:
        print("No response from Gemini after retry, returning empty result.")
        return pd.DataFrame()

    result = EventList.model_validate(response)

    df = pd.DataFrame([e.model_dump() for e in result.events])

    if not df.empty:
        df.to_excel(writer_raw, sheet_name=company, index=False)
    else:
        print(f"No data to write for {company}")

    df_metrics = pd.DataFrame()

    if result and result.events:
        print(f"Validating {len(result.events)} extracted events from {company}...")
        df_metrics = batched_validate_output(search_chunks, result, 5)

    else:
        print("No events found in the response.")

    return df_metrics
