from google import genai
import adtiam

# adtiam.load_creds("utest")

client = genai.Client(api_key="AIzaSyCkLBPubQBiXS8XHNrRalS-jnTfzSozTGQ")

model = client.models.get(model="gemini-2.0-flash")
print(f"{model.input_token_limit=}")
print(f"{model.output_token_limit=}")

model = client.models.get(model="gemini-1.5-flash")
print(f"{model.input_token_limit=}")
print(f"{model.output_token_limit=}")
