from pydantic import Field, BaseModel
from typing import Dict, List, Optional


class EventCatalyst(BaseModel):
    company: str = Field(
        description="The name of the company, example: Taysha Gene Therapies"
    )
    accession_number: str = Field(
        description="The SEC Accession Number that uniquely identifies a specific filing submission, e.g., 0001689548-23-000044 "
    )
    drug: str = Field(
        description="The name of the drug being studied, example: ulixacaltamide"
    )
    program: str = Field(
        description="The clinical program or indication the drug is being developed for, example: ENERGY Program"
    )
    phase: str = Field(
        description="The phase of the clinical study. do not add study name. Example: Phase 2a"
    )
    study: str = Field(
        description="The specific clinical study identifier or name. do not add phase. example: Phase 2a Photo-Paroxysmal Response study is Photo-Paroxysmal Response study"
    )
    size: str = Field(
        description="The number of patients enrolled in the study, example: 1000"
    )
    status_announce: str = Field(
        description="Whether topline results from the study were already announced or still expected. Make A=Actual, E=Expected, example: A"
    )
    time_period_expected: str = Field(
        description="The time period when topline results from the study were announced or are expected. State year then quarter/month etc, example: 2025Q2, 2025H2"
    )
    explanation: str = Field(
        description="Summarize additional context or explanation about the study results"
    )
    primary_endpoint_result: str = Field(
        description="Outcome of the primary efficacy endpoint, example: Met, Not Met, Partially Met"
    )
    adverse_events_summary: str = Field(
        description="Summary of observed adverse events, including severity and incidence, example: Low incidence of SAEs; mild TEAEs"
    )
    regulatory_milestone: str = Field(
        description="Latest regulatory development or designation for the drug, example: NDA Submitted, Fast Track Granted"
    )
    secondary_endpoint_notes: str = Field(
        description="Optional notes on notable secondary endpoint results, if disclosed"
    )
    trial_design: str = Field(
        description="Optional description of trial methodology, such as double-blind, placebo-controlled"
    )
    biomarkers_used: str = Field(
        description="Optional note on whether biomarkers were used in the trial for selection, monitoring, or endpoints"
    )
    comparator_used: str = Field(
        description="Optional description of the control or comparator arm, example: Placebo, Standard of Care"
    )
    geography: str = Field(
        description="Regions where trial or regulatory submission occurred or where marketing authorization is being sought"
    )
    submission_type: str = Field(
        description="Type of regulatory submission, example: NDA, MAA, BLA, resubmission, Type II variation"
    )
    regulatory_track: str = Field(
        description="Review designation or track assigned by regulatory body, example: Priority Review, Fast Track, Accelerated Approval"
    )
    milestone_trigger: str = Field(
        description="Indicates if the event triggered a financial milestone or partnership payment, example: Novartis $1B upfront payment"
    )
    clinical_benefit_summary: str = Field(
        description="Narrative on overall perceived benefit or trend, e.g., dose-dependent HTT lowering, improvement in stability subscale"
    )
    readout_type: str = Field(
        description="The type of results expected or announced, e.g., Topline, Interim, Final, Long-term Follow-up"
    )
    trial_status: str = Field(
        description="Current operational status of the trial, example: Enrolling, Dosing, Completed, Terminated"
    )


class EventList(BaseModel):
    events: List[EventCatalyst]


class ValidationFeedback(BaseModel):
    is_accurate: bool = Field(
        description="True if the extracted data is fully accurate and complete according to the original text, False otherwise."
    )
    corrected_data: Optional[EventList] = Field(
        None,
        description="The corrected EventList object if inaccuracies were found and corrected. "
        "Provide the full, corrected EventList structure here. If no corrections are needed, this field should be null.",
    )
