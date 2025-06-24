import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from schema import EventList, EventCatalyst
from langchain_openai import ChatOpenAI
import adtiam
import os
from typing import Dict
from google import genai
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
adtiam.load_creds("adt-llm")
os.environ["OPENAI_API_KEY"] = adtiam.creds["llm"]["openai"]


def extract_kpi(search_metric, search_chunks, metadata: Dict | None = None):
    today = datetime.now()
    formatted_date = today.strftime("%Y-%m-%d")

    company = "not specified"
    accession_number = "not specified"
    if metadata:
        company = metadata.get("company", "not specified")
        accession_number = metadata.get("accession_number", "not specified")

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
- IMPORTANT: **Return one object per distinct drug development program or major clinical trial ONLY IF that program or trial is directly relevant to (strictly requiring an explicit mention or a very strong, unambiguous contextual link), or explicitly mentions, the metrics provided in `{search_metric}`.** For each such relevant program/trial, a "distinct" object is warranted for each:
    - **Unique drug candidate** (a different molecule, biologic, or therapeutic modality).
    - **New disease indication** being pursued for an existing drug candidate.
    - **Major, separately-named clinical trial** (e.g., a pivotal proof-of-concept, Phase 2b, or Phase 3 study) even for an existing drug-indication pair, as these represent discrete, high-impact data readouts and potential value inflection points.
- Be concise and consistent, but thorough — extract everything real, nothing imagined.
- Treat strings as case-insensitive when comparing or grouping values (e.g., "Phase 2a" and "phase 2A" are the same).

[General Instructions]
accession_number fallback value: {accession_number} -> Use only if the SEC Accession Number is not explicitly stated in the text.
**ABSOLUTE PRIORITY RULE: For any given clinical trial or event that is relevant to the search metric, ALWAYS extract and return the LATEST, MOST CURRENT, and DEFINITIVELY CONFIRMED information. If a result or status was once 'expected' but the text later states it 'was announced' or 'demonstrated' a specific outcome, you MUST use the announced/demonstrated status and result.**
Crucial Rule: ONLY use 'not specified' for a field if the information is genuinely ABSENT from the provided text for that specific event. Do NOT use 'not specified' if the information can be reasonably inferred or is directly stated elsewhere in the text for that entry
Current Date: {formatted_date}

[schema]
EventCatalyst:
company: The exact, full name of the company from text (e.g., 'Taysha Gene Therapies, Inc.'). If not found in text, use '{company}'; if '{company}' is also empty, use 'not specified'. Do not confuse this field with a ticker symbol.
accession_number: The SEC Accession Number that uniquely identifies a specific filing submission, e.g., 0001689548-23-000044 
drug: Name of the drug, e.g., ulixacaltamide  
program: The indication or program, e.g., ENERGY Program  
phase: Clinical phase only (exclude study name), e.g., Phase 2a  
study: Study identifier or title only (exclude phase), e.g., Photo-Paroxysmal Response study  
size: Number of enrolled patients, e.g., 1000  
status_announce: Use 'A' (Actual/Announced) ONLY if the results/event have been definitively announced (e.g., 'results were announced', 'demonstrated positive results'). Use 'E' (Expected) if the event is still anticipated in the future relative to 'Current Date' (e.g., 'expected in Q3 2025'). **If an event was 'expected' at a date BEFORE 'Current Date' AND no 'Actual/Announced' result is found in the text, maintain 'E' as the status (indicating an overdue expectation).** Use 'P' (Planned) if the trial is described as being planned or will be initiated. Default to 'not specified' if no clear status is mentioned.
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
    # llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

    # structured_llm = llm.with_structured_output(EventList)

    # result: EventList = EventList.model_validate(structured_llm.invoke(llm_prompt))

    # df_metrics = pd.DataFrame([e.model_dump() for e in result.events])
    client = genai.Client(api_key=os.getenv("google_api_key"))

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=llm_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": EventList.model_json_schema(),
        },
    )
    result = None
    df_metrics = pd.DataFrame()

    if response.text:
        response_json = json.loads(response.text)
    if response_json:
        result = EventList.model_validate(response_json)
        df_metrics = pd.DataFrame([e.model_dump() for e in result.events])
    return df_metrics
