import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from schema import EventList, EventCatalyst
from langchain_openai import ChatOpenAI
import adtiam
import os
from typing import Dict

adtiam.load_creds("adt-llm")
os.environ["OPENAI_API_KEY"] = adtiam.creds["llm"]["openai"]


def extract_kpi(search_metric, search_chunks, metadata: Dict | None = None):
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
- Return one object **per distinct drug program or clinical trial** described.
- Be concise and consistent, but thorough — extract everything real, nothing imagined.
- Treat strings as case-insensitive when comparing or grouping values (e.g., "Phase 2a" and "phase 2A" are the same).

[schema]
EventCatalyst:
company: The name of the company / The exact name of Registrant as specified in its charter. 
Company name: {company} -> If not specified in text, fill this field from given data or leave as 'not specified'.
accession_number: The SEC Accession Number that uniquely identifies a specific filing submission, e.g., 0001689548-23-000044 
accession_number: {accession_number} -> If not specified in text, fill this field from given data or leave as 'not specified'.
drug: Name of the drug, e.g., ulixacaltamide  
program: The indication or program, e.g., ENERGY Program  
phase: Clinical phase only (exclude study name), e.g., Phase 2a  
study: Study identifier or title only (exclude phase), e.g., Photo-Paroxysmal Response study  
size: Number of enrolled patients, e.g., 1000  
status_announce: 'A' if actual/topline announced, 'E' if expected, e.g., A  
time_period_expected: Time results were announced or are expected, e.g., 2025Q2, 2025H2  
explanation: Summary or context on the trial results or disclosure  
primary_endpoint_result: Result of primary efficacy endpoint, e.g., Met, Not Met  
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
trial_status: Operational status of trial, e.g., Enrolling, Completed, Terminated

EventList:
events: List of EventCatalyst objects

[data]
{" ".join(search_chunks)}

[metrics]
Extract and return all relevant clinical, trial, and regulatory information from the above data that mentions the metrics below:
{search_metric}


Output the result as a JSON list of `EventCatalyst` objects, like:
[
  EventCatalyst(...),
  EventCatalyst(...)
]
"""
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

    structured_llm = llm.with_structured_output(EventList)

    result: EventList = EventList.model_validate(structured_llm.invoke(llm_prompt))

    df_metrics = pd.DataFrame([e.model_dump() for e in result.events])

    return df_metrics
