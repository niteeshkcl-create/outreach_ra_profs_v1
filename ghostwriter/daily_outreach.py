import csv
import os
import json
import time
import base64
from datetime import datetime
import google.generativeai as genai
import google.auth
# Gmail API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.message import EmailMessage
import re
import requests

# Configuration
ALLEN_CSV = "data/allen_faculty_all.csv"
ESCIENCE_CSV = "data/escience_faculty_all.csv"
SENT_LOG_PATH = "data/sent_log.csv"
FAILED_LOG_PATH = "data/failed_outreach.csv"
RESUMES_EXTRACTED_JSON = "data/resumes_extracted.json"
TOKEN_PATH = 'token.json'
CREDENTIALS_PATH = 'credentials.json'
NOTIFICATION_EMAIL = "niteesh.k.cl@gmail.com"
DAILY_LIMIT_PER_SOURCE = 7
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Initialize Gemini
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-flash-latest") # The identifier that actually works
else:
    print("Warning: GEMINI_API_KEY not found. LLM features will be disabled.")
    model = None

def call_ollama(prompt):
    """Backup call to local Ollama (Llama3)."""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=90) # Increased timeout
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"Ollama error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

GEMINI_EXHAUSTED = False

def call_gemini_with_retry(prompt, max_retries=3, initial_delay=10):
    global GEMINI_EXHAUSTED
    if model and not GEMINI_EXHAUSTED:
        for i in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):
                    GEMINI_EXHAUSTED = True
                    print(f"Gemini quota exceeded. Falling back to Ollama for the rest of the session.")
                    break
                else:
                    print(f"Gemini API error: {e}")
                    break
    
    # Fallback to Ollama if Gemini fails or is disabled
    return call_ollama(prompt)

def extract_email_from_text(text):
    if not text:
        return None
    # Look for common email patterns, prioritizing @cs.uw.edu or @uw.edu
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.edu', text)
    if emails:
        # Prioritize UW emails
        for e in emails:
            if 'uw.edu' in e.lower():
                return e.lower()
        return emails[0].lower()
    return None

def load_processed_names():
    processed = set()
    # Load successful sends
    if os.path.exists(SENT_LOG_PATH):
        with open(SENT_LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            processed.update({row["name"] for row in reader})
    
    # Load today's failures to skip them in the same run
    if os.path.exists(FAILED_LOG_PATH):
        with open(FAILED_LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            today = datetime.now().strftime("%Y-%m-%d")
            for row in reader:
                if row["date"] == today:
                    processed.add(row["name"])
    return processed

def update_sent_log(name, email):
    with open(SENT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, email, datetime.now().strftime("%Y-%m-%d")])

def log_failed_outreach(name, email, bio_link, reason):
    if not os.path.exists(FAILED_LOG_PATH):
        with open(FAILED_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "email", "bio_link", "reason", "date"])
    
    with open(FAILED_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, email, bio_link, reason, datetime.now().strftime("%Y-%m-%d")])

def load_faculty(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_resumes():
    if not os.path.exists(RESUMES_EXTRACTED_JSON):
        return {}
    with open(RESUMES_EXTRACTED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def select_best_resume(prof_bio, resumes_dict):
    nlp_fallback = "niteesh_ds_as_mle_role_all_nlp.pdf"
    if not model:
        return nlp_fallback if nlp_fallback in resumes_dict else next(iter(resumes_dict.keys()))
    
    resume_summaries = ""
    for name, text in resumes_dict.items():
        # Using more of the resume text for better matching
        resume_summaries += f"Filename: {name}\nSnippet: {text[:1500]}\n---\n"
    
    prompt = f"""
    You are an expert academic advisor helping a graduate student find a Research Assistant (RA) position at the University of Washington.
    Compare the following resumes and select THE BEST ONE that fits the professor's research interests and background.
    
    MATCHING STRATEGY:
    1. PRIORITIZE TECHNICAL DEPTH: Match projects and skills (NLP, Systems, ML, etc.) to the professor's expertise.
    2. THE "ALL_NLP" PREFERENCE: The student considers the 'niteesh_ds_as_mle_role_all_nlp.pdf' resume to be their most comprehensive and strong technical resume. If it matches the professor's technical domain decently, prefer it over more niche versions.
    3. ALIGNMENT:
       - If the professor mentions 'NLP', 'LLMs', or 'Language', definitely pick an NLP-focused resume.
       - If the professor mentions 'Systems', 'Networking', or 'P4', look for systems projects in resumes.
       - If the professor is a 'Teaching Professor' or mentions 'Pedagogy', ensure the selected resume includes the 'Teaching Assistant' experience (available in most).
    4. FALLBACK: If nothing matches well, or you are unsure, return '{nlp_fallback}'.
    
    Professor's Bio:
    {prof_bio[:1500]}
    
    Resumes (Filename & Content):
    {resume_summaries}
    
    Return ONLY the filename of the best-fitting resume. No explanation.
    """
    try:
        content = call_gemini_with_retry(prompt)
        if not content:
            return nlp_fallback if nlp_fallback in resumes_dict else next(iter(resumes_dict.keys()))
            
        selected = content.replace('"', '').replace("'", "")
        
        # Exact match check
        if selected in resumes_dict:
            return selected
            
        # Fuzzy/Contain match check
        for key in resumes_dict.keys():
            if key in selected or selected in key:
                return key
                
        return nlp_fallback if nlp_fallback in resumes_dict else next(iter(resumes_dict.keys()))
    except Exception as e:
        print(f"Error selecting best resume: {e}")
        return nlp_fallback if nlp_fallback in resumes_dict else next(iter(resumes_dict.keys()))

def draft_cover_letter(prof, resume_text):
    if not model:
        return "Generic Subject", "Generic Body"
    
    # Context for drafting
    source_context = "Allen School" if "Allen" in prof.get("source", "") else "eScience Institute"
    
    prompt = f"""
    You are an ambitious student reaching out to a professor for research opportunities. 
    Using my resume content and the professor's research bio, draft a personalized, professional cover letter email.
    The email should clearly demonstrate how my skills align with their current research.
    
    Professor: {prof['name']} at UW {source_context}
    Professor's Bio: {prof.get('bio', '')[:1000]}
    My Resume Content: {resume_text[:2000]}
    
    Return your response ONLY in JSON format with 'subject' and 'body' fields. 
    Ensure the tone is respectful and concise. Use my name 'Niteesh' in the signature.
    """
    try:
        content = call_gemini_with_retry(prompt)
        if not content:
            return None, None
            
        # Cleanup markdown formatting and non-JSON text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Robust JSON cleaning: handle unescaped control characters from some LLMs
        def clean_json_string(s):
            # Remove everything before the first { and after the last }
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1:
                s = s[start:end+1]
            # Remove control characters except for newline, carriage return, and tab
            return "".join(char for char in s if ord(char) >= 32 or char in "\n\r\t")

        cleaned_content = clean_json_string(content)
        
        try:
            data = json.loads(cleaned_content)
        except json.JSONDecodeError:
            # Last ditch effort: regex for subject and body
            subject_match = re.search(r'"subject":\s*"(.*?)"', cleaned_content, re.DOTALL)
            body_match = re.search(r'"body":\s*"(.*?)"', cleaned_content, re.DOTALL)
            if subject_match and body_match:
                data = {
                    "subject": subject_match.group(1).replace("\\n", "\n").replace('\\"', '"'),
                    "body": body_match.group(1).replace("\\n", "\n").replace('\\"', '"')
                }
            else:
                raise
            
        # Final validation to prevent generic sends
        if "Dear Prof." in data["body"] and "I'm interested in your work" in data["body"] and len(data["body"]) < 100:
            return None, None
            
        return data["subject"], data["body"]
    except Exception as e:
        print(f"Error drafting cover letter for {prof['name']}: {e}")
        return None, None

def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def send_email(service, to_email, subject, body, attachment_path=None, max_retries=3):
    """Sends an email with optional attachment and retry logic."""
    for i in range(max_retries):
        try:
            message = EmailMessage()
            message.set_content(body)
            message['To'] = to_email
            message['Subject'] = subject

            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(attachment_path)
                message.add_attachment(
                    file_data,
                    maintype='application',
                    subtype='pdf',
                    filename=file_name
                )

            # encoded message
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

            create_message = {
                'raw': encoded_message
            }
            
            send_message = (service.users().messages().send(userId="me", body=create_message).execute())
            print(f'Email successfully sent to {to_email}. Message Id: {send_message["id"]}')
            return True
        except Exception as e:
            if i < max_retries - 1:
                wait_time = 5 * (i + 1)
                print(f"Transient error sending to {to_email}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Final failure sending to {to_email} after {max_retries} attempts: {e}")
                return False

def daily_outreach(dry_run=True, test_email_only=None, limit_override=None):
    sent_names = load_processed_names()
    allen_faculty = load_faculty(ALLEN_CSV)
    escience_faculty = load_faculty(ESCIENCE_CSV)
    resumes_dict = load_resumes()
    
    today_success_count = 0
    if os.path.exists(SENT_LOG_PATH):
        with open(SENT_LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            today = datetime.now().strftime("%Y-%m-%d")
            for row in reader:
                if row.get("date_sent") == today:
                    today_success_count += 1
    
    if limit_override:
         target_successes_needed = limit_override
         print(f"Manual override: Sending {limit_override} more emails regardless of daily quota.")
    else:
         target_successes_needed = 14 - today_success_count
         print(f"Total successes today: {today_success_count}. Target: 14. Need {target_successes_needed} more.")
         
         if target_successes_needed <= 0:
            print("Today's target already reached!")
    
    service = None
    if not dry_run or test_email_only:
        print("Authenticating with Gmail...")
        service = get_gmail_service()
    
    # Notify start of run (always send if live or test-notify)
    is_scheduled_run = not dry_run and not test_email_only
    force_notify = hasattr(sys, '_test_notify')
    
    if is_scheduled_run or force_notify:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        goal_msg = f"Goal: Send {limit_override} more emails." if limit_override else "Goal: Reach a total of 14 successes today."
        
        send_email(
            service, 
            NOTIFICATION_EMAIL, 
            f"Outreach Started: {datetime.now().strftime('%Y-%m-%d')}", 
            f"The daily faculty outreach process started at {start_time}.\n{goal_msg}\nCurrent successes: {today_success_count}.\nNeed {max(0, target_successes_needed)} more."
        )

    if not limit_override and target_successes_needed <= 0:
        print("Today's target already reached!")
        if is_scheduled_run or force_notify:
            # Send final report immediately since no work needed
            send_email(
                service, 
                NOTIFICATION_EMAIL, 
                f"Outreach Completed (Already Done): {datetime.now().strftime('%Y-%m-%d')}", 
                f"The daily faculty outreach process checked at {datetime.now().strftime('%H:%M:%S')} and found the target of 14 emails was already reached for today.",
                attachment_path=SENT_LOG_PATH
            )
        return

    if not resumes_dict:
        print("Error: No resumes found in data/resumes_extracted.json. Run extract_resumes.py first.")
        return
    to_send = []
    
    if test_email_only:
        # Just pick one professor for a live test
        if allen_faculty: to_send = [allen_faculty[0]]
        else: to_send = [escience_faculty[0]]
        print(f"TEST MODE: Sending 1 email to {test_email_only} for {to_send[0]['name']}")
    else:
        # Prioritize Allen Faculty as per user's usual preference (after Leilani Battle)
        # We'll gather enough candidates to hit the target, allowing for some personalization failures
        candidate_pool_size = target_successes_needed * 2 
        
        count = 0
        for prof in allen_faculty:
            if count >= candidate_pool_size: break
            if prof["name"] not in sent_names:
                to_send.append(prof)
                count += 1
                
        # If still need more, take from eScience
        if len(to_send) < candidate_pool_size:
            for prof in escience_faculty:
                if count >= candidate_pool_size: break
                if prof["name"] not in sent_names:
                    # Skip fellows if they look like students (bio contains "graduate student" or "PhD student")
                    bio = prof.get("bio", "").lower()
                    if "graduate student" in bio or "phd student" in bio or "doctoral student" in bio:
                        continue
                    to_send.append(prof)
                    count += 1
            
    print(f"Processing up to {len(to_send)} outreach targets to reach target of 14 successes...")
    if not dry_run:
        print("Wait 2 mins before starting to clear possible rate limits...")
        time.sleep(120)
    
    daily_drafts = []
    current_success_today = today_success_count
    # If limit_override is set, we track sends in this session only for the stopping condition
    session_sends = 0
    
    for prof in to_send:
        if limit_override:
             if session_sends >= limit_override:
                 print(f"Reached manual limit of {limit_override} sends for this session!")
                 break
        elif current_success_today >= 14:
            print("Reached target of 14 successful sends for today!")
            break
        # 1. First, check for a valid email before doing anything else
        raw_email = prof.get("email", "").strip()
        bio_email = extract_email_from_text(prof.get("bio", ""))
        
        target_email = None
        if test_email_only:
            target_email = test_email_only
        elif raw_email and "@" in raw_email:
            target_email = raw_email
        elif bio_email:
            target_email = bio_email
            
        if not target_email or target_email == "placeholder@uw.edu":
            # Logging skip to failed_outreach as requested
            log_failed_outreach(prof['name'], target_email or "None", prof.get("profile_link"), "No valid email found")
            continue

        # 2. Proceed with drafting only if email is found
        print(f"\nMatching and drafting for {prof['name']} ({target_email})...")
        best_resume_name = select_best_resume(prof.get("bio", ""), resumes_dict)
        resume_text = resumes_dict[best_resume_name]
        
        subject, body = draft_cover_letter(prof, resume_text)
        
        if not subject or not body:
            print(f"Skipping {prof['name']}: Personalization failed (LLM error).")
            log_failed_outreach(prof['name'], target_email, prof.get("profile_link"), "Personalization Failed")
            continue

        if service:
             resume_path = os.path.join("resume", best_resume_name)
             success = send_email(service, target_email, subject, body, attachment_path=resume_path)
             if success and not test_email_only:
                 update_sent_log(prof["name"], target_email)
                 current_success_today += 1
                 session_sends += 1
             elif not success:
                 print(f"Failed to send email to {target_email} after retries.")
                 log_failed_outreach(prof['name'], target_email, prof.get("profile_link"), "Gmail API Send Failure")
        else:
            print(f"Dry run: Skipping email to {target_email}")
        
        draft_record = {
            "name": prof["name"],
            "email": target_email,
            "resume_used": best_resume_name,
            "subject": subject,
            "body": body,
            "profile_link": prof.get("profile_link"),
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        daily_drafts.append(draft_record)
        
        # Delay between emails
        time.sleep(120)

    # Save daily results to a file for user review
    batch_filename = f"outreach_results_{datetime.now().strftime('%Y%m%d')}.json"
    with open(f"data/{batch_filename}", "w", encoding="utf-8") as f:
        json.dump(daily_drafts, f, indent=4)
        
    print(f"\nCompleted! {len(daily_drafts)} results saved to data/{batch_filename}")

    # Notify end of run (only if we did work)
    if (not dry_run and not test_email_only) or hasattr(sys, '_test_notify'):
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_successes = current_success_today - today_success_count
        send_email(
            service, 
            NOTIFICATION_EMAIL, 
            f"Outreach Completed: {datetime.now().strftime('%Y-%m-%d')}", 
            f"The daily faculty outreach process completed at {end_time}.\nTotal successful sends today: {current_success_today}.\nNew sends in this session: {new_successes}.\n\nAttached is the current sent log.",
            attachment_path=SENT_LOG_PATH
        )

if __name__ == "__main__":
    import sys
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-notify":
        print("Testing start/end notifications...")
        sys._test_notify = True
        daily_outreach(dry_run=False, test_email_only="niteesh.k.cl@gmail.com")
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        daily_outreach(dry_run=False, test_email_only="niteesh.k.cl@gmail.com")
    elif len(sys.argv) > 1 and sys.argv[1] == "--live":
        if len(sys.argv) > 3 and sys.argv[2] == "--limit":
             limit = int(sys.argv[3])
             daily_outreach(dry_run=False, limit_override=limit)
        else:
             daily_outreach(dry_run=False)
    else:
        print("Running in DRY RUN mode. Use --live for actual sending or --test for one email.")
        daily_outreach(dry_run=True)
