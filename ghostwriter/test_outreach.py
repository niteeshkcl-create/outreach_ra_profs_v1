import csv
import os
import json
import time
from datetime import datetime
import google.generativeai as genai
import sys

# Add the project root to path so we can import from ghostwriter if needed, 
# but for simplicity I'll copy the core logic here with the test override.

SENT_LOG_PATH = "data/sent_log.csv"
ALLEN_CSV = "data/allen_faculty_all.csv"
ESCIENCE_CSV = "data/escience_faculty_all.csv"
RESUMES_EXTRACTED_JSON = "data/resumes_extracted.json"
TEST_EMAIL = "niteesh.k.cl@gmail.com"

# Initialize Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-flash-latest")

def load_faculty(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_resumes():
    with open(RESUMES_EXTRACTED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def select_best_resume(prof_bio, resumes_dict):
    resume_options = "\n".join([f"- {name}: {text[:300]}..." for name, text in resumes_dict.items()])
    prompt = f"""
    You are an expert career counselor. Given the following professor's research bio and a list of my resumes, 
    select the ONE resume filename that best matches the professor's research interests.
    
    Professor's Bio:
    {prof_bio[:1000]}
    
    My Resume Options:
    {resume_options}
    
    Return ONLY the filename of the best matching resume.
    """
    response = model.generate_content(prompt)
    selected = response.text.strip()
    for key in resumes_dict.keys():
        if key in selected:
            return key
    return next(iter(resumes_dict.keys()))

def draft_cover_letter(prof, resume_text):
    source_context = "Allen School" if "Allen" in prof["source"] else "eScience Institute"
    prompt = f"""
    You are an ambitious student reaching out to a professor for research opportunities. 
    Using my resume content and the professor's research bio, draft a personalized, professional cover letter email.
    
    Professor: {prof['name']} at UW {source_context}
    Professor's Bio: {prof.get('bio', '')[:1000]}
    My Resume Content: {resume_text[:2000]}
    
    Return your response in JSON format with 'subject' and 'body' fields. 
    Ensure the tone is respectful and concise. Use [Your Name] as placeholder.
    """
    response = model.generate_content(prompt)
    content = response.text.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    data = json.loads(content)
    return data["subject"], data["body"]

def run_single_test():
    allen_faculty = load_faculty(ALLEN_CSV)
    resumes_dict = load_resumes()
    
    if not allen_faculty:
        print("No faculty found.")
        return

    # Pick the first professor for the test
    prof = allen_faculty[0]
    print(f"--- TEST RUN FOR {prof['name']} ---")
    
    print("Matching best resume...")
    best_resume_name = select_best_resume(prof.get("bio", ""), resumes_dict)
    print(f"Selected Resume: {best_resume_name}")
    
    print("Drafting personalized cover letter...")
    subject, body = draft_cover_letter(prof, resumes_dict[best_resume_name])
    
    print(f"\nTARGET EMAIL (OVERRIDDEN): {TEST_EMAIL}")
    print(f"SUBJECT: {subject}")
    print("\nBODY:")
    print(body)
    print("\n--- TEST RUN COMPLETE ---")

if __name__ == "__main__":
    run_single_test()
