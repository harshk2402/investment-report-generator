import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from schema import EventList, EventCatalyst
from langchain_openai import ChatOpenAI
import adtiam
import os

adtiam.load_creds("adt-llm")
os.environ["OPENAI_API_KEY"] = adtiam.creds["llm"]["openai"]


def extract_kpi(search_metric, search_chunks, metadata=None):

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

[schema]
EventCatalyst:
company: The name of the company / The exact name of Registrant as specified in its charter, e.g., Taysha Gene Therapies  
cik: The SEC Central Index Key (CIK), e.g., 0001070081  
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
    df_metrics.sort_values(by="company")

    return df_metrics


# metric_name: states the name of the kpi to extracted. this should be as comprehensive as possible and take into account any hierarchical structure in a table or so where the name is split across different lines or grouped together.
# metric_type: describes the type of metric, for example: growth_yoy for year over year growth, growth_qoq for quarter over quarter sequential growth, currency for dollar or other currency amounts, amount for a count or other numeric amount
# metric_unit: extract the units for the table, for example m for millions, $m for $ in millions, % for percentage. Note that this information is often contained in the header or footer of the table.
# time_period: describes the time period of the metric, for example: 2024Q1 for the first quarter of 2024, FY2024 for the year 2024, FYQ1 for the first quarter when year is not given
# metric_value: the value of the metric, for example: 9% for 9 percent growth, 1000000 for 1 million dollars, 1000000000 for 1 billion amount
# explanation: extract commentary about this number, for example why it has changed. this might be contained in a footnote to the table.

# example:
# {search_metric} | global room nights Europe | growth_yoy | % | 2024Q1 | 9% | due to higher retail pricing, partially offset by a higher mix of subscribers to wholesale offerings
# {search_metric} | global room nights Americs | growth_yoy | % | 2024Q1 | 9% | Null
# {search_metric} | global room nights Europe | growth_yoy | % | 2023Q1 | 9% | Null
# {search_metric} | global room nights Americs | growth_yoy | % | 2023Q1 | 9% | Null
# {search_metric} | Average Daily Rate Europe | currency | $ | 2024Q1 | $7.18 | due to higher retail pricing, partially offset by a higher mix of subscribers to wholesale offerings
# {search_metric} | Average Daily Rate Americs | currency | $ | 2024Q1 | $7.18 | Null
# {search_metric} | Average Daily Rate Europe | currency | $ | 2023Q1 | $7.18 | Null
# {search_metric} | Average Daily Rate Americs | currency | $ | 2023Q1 | $7.18 | Null

# extract all values the best you can do with the information provided. Fill with "Null" when one of the fields cannot be extracted.


# [metrics]
# {search_metric}

# [format]
# {' | '.join(KPIMetric.model_fields)}

# return only the data in a pipe delimited string and nothing else.
# Initialize OpenAI client
# client = OpenAI(api_key=o3studio.settings.creds["openai"])
# Make API call
# completion = client.beta.chat.completions.parse(
# completion = client.chat.completions.create(
#     model="gpt-4o-2024-08-06",
#     messages=[
#         {
#             "role": "system",
#             "content": "Extract the KPI metric information from the provided text.",
#         },
#         {"role": "user", "content": llm_prompt},
#     ],
#     temperature=0,
#     response_format=EventList,
# )

# Get the parsed result
# result = completion.choices[0].message.content.strip()

# Split the result into lines and create DataFrame
# lines = [line.strip() for line in result.split("\n") if line.strip()]
# df_metrics = pd.DataFrame(
#     [line.split("|") for line in lines], columns=list(EventCatalyst.model_fields)
# )


# class KPIMetric(BaseModel):
#     search_metric: str
#     metric_name: str
#     metric_type: str
#     metric_unit: str
#     time_period: str
#     metric_value: str
#     explanation: str
