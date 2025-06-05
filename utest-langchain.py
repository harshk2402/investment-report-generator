import langchain
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import adtiam
from schema import EventList

import cfg

# Initialize the OpenAI chat model
llm = ChatOpenAI(api_key=adtiam.creds["llm"]["openai"], model_name="gpt-3.5-turbo")

# Create the output parser
parser = PydanticOutputParser(pydantic_object=EventList)

data = {}

# Create a message with structured output instructions
structured_prompt = f"""
"You are a biomedical research analyst. Your task is to extract and structure all clinically and regulatorily relevant "
                "drug development and trial data from the following biotech disclosure text (e.g., financial filings, press releases).\n\n"

                "You must output a JSON list of objects that match the exact schema below. The output must:\n"
                "- Include **all** fields listed in the schema — do not omit any field under any circumstance.\n"
                "- Use the **exact field names** as defined in the schema.\n"
                "- Populate fields only if the information is explicitly mentioned or clearly implied by the text.\n"
                "- If the information is **not available**, return the string `'not specified'` for that field (not null, empty, or omitted).\n"
                "- Do **not hallucinate** or infer beyond the given text.\n"
                "- Do **not duplicate entries** or fields.\n"
                "- Be concise and structured, yet thorough — extract everything relevant but nothing invented.\n"
                "- If multiple distinct drug programs or trials are described, return one object per program/trial in the list.\n\n"
{parser.get_format_instructions()}

Text: {data}
"""

messages = [HumanMessage(content=structured_prompt)]

# Get the response
response = llm.invoke(messages)

# Parse the response into structured format
try:
    parsed_output = parser.parse(response.content)
    print("\nStructured Output:")
    print(f"Amount: ${parsed_output.amount}M")
    print(f"Quarter: {parsed_output.quarter}")
    print(f"Year: {parsed_output.year}")
except Exception as e:
    print(f"Error parsing response: {e}")
    print("Raw response:", response.content)
