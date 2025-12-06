import requests
import json
import re

class LabReportExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
        
        self.features = [
            "Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"
        ]

    def build_prompt(self, report_text):
        return f"""
You are a medical data extraction AI. Extract the following fields from the lab report text.

Units and conversions required:
- Extract all metabolic values in **mmol/L**
- **Creatinine (Cr) must be returned in µmol/L**
- **Do NOT convert Gender, Age, or BMI**
- If a value is already in the correct unit, keep it unchanged
- If a value is in mg/dL, convert to mmol/L using:
  - Urea (mg/dL → mmol/L): value × 0.1665
  - Chol, HDL, LDL, VLDL (mg/dL → mmol/L): value × 0.0259
  - Triglycerides TG (mg/dL → mmol/L): value × 0.0113
- Creatinine: if reported in mg/dL convert to µmol/L using: value × 88.4

Gender format:
- Male = 1
- Female = 0

If a value is missing, return null.

Return ONLY a valid JSON object. No text before or after the JSON.

Lab report:
{report_text}

JSON:
{{
  "Gender": <int>,
  "AGE": <number or null>,
  "Urea": <number or null>,
  "Cr": <number or null>,
  "HbA1c": <number or null>,
  "Chol": <number or null>,
  "TG": <number or null>,
  "HDL": <number or null>,
  "LDL": <number or null>,
  "VLDL": <number or null>,
  "BMI": <number or null>
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
