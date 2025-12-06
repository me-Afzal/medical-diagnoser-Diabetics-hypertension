import requests
import json
import re

class LabReportExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
        
        # Hypertension dataset features for output
        self.features = [
            "Age", "Salt_Intake", "Stress_Score", "BP_History", "Sleep_Duration",
            "BMI", "Medication", "Family_History", "Exercise_Level", "Smoking_Status"
        ]

    def build_prompt(self, report_text):
        return f"""
Extract the following fields from the medical text:

Age (number),
Salt_Intake (grams/day),
Stress_Score (0–10),
BP_History (Normal / Prehypertension / Hypertension),
Sleep_Duration (hours/day),
BMI (kg/m²),
Medication (name, or "No" if none),
Family_History (Yes/No),
Exercise_Level (Low / Moderate / High),
Smoking_Status (Non-Smoker / Smoker)

If a value is missing, return null.
Return ONLY a valid JSON object. No text before or after.

Medical text:
{report_text}

JSON:
{{
  "Age": <number or null>,
  "Salt_Intake": <number or null>,
  "Stress_Score": <number or null>,
  "BP_History": <string or null>,
  "Sleep_Duration": <number or null>,
  "BMI": <number or null>,
  "Medication": <string or null>,
  "Family_History": <string or null>,
  "Exercise_Level": <string or null>,
  "Smoking_Status": <string or null>
}}
"""

    def extract(self, report_text):
        headers = {"Content-Type": "application/json", "X-goog-api-key": self.api_key}
        payload = {"contents": [{"parts": [{"text": self.build_prompt(report_text)}]}]}

        response = requests.post(self.url,
                                     headers=headers,
                                     json=payload,
                                     timeout=20)

        if response.status_code != 200:
            raise Exception(f"Gemini API Error: {response.text}")

        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]

        # convert string JSON to dictionary
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            json_str = match.group(0)
            data = json.loads(json_str)
        except json.JSONDecodeError:
            raise Exception("LLM returned non-JSON format")

        # Validation: keep only desired fields
        cleaned = {k: data.get(k, None) for k in self.features}
        return cleaned
