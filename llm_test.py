import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain what a budget is to a 13-year-old in 3 sentences.",
)
print(response.text)