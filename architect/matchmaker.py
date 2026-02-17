import json
import os
from typing import List, Dict

# Note: In a real scenario, we'd use gemini embeddings or similar.
# For this agentic implementation, we will use a dedicated "Matchmaker" logic
# that leverages the LLM's reasoning to score the match.

def load_data(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def matchmaker():
    profiles = load_data("data/faculty_deep_profiles.json")
    
    # We expect these resumes to be provided as text/PDFs
    # For now, let's assume we have the descriptions of the 3 resumes
    resumes = {
        "A": "Machine Learning: Expert in deep learning, neural networks, and computer vision. Experience with PyTorch and large-scale model training.",
        "B": "Systems: Focused on distributed systems, operating systems, and high-performance computing. Experience with kernels and cloud infrastructure.",
        "C": "Data Science: Specialist in statistical analysis, data visualization, and interdisciplinary data applications. Proficient in R and big data tools."
    }

    print(f"Ranking {len(profiles)} profiles against {len(resumes)} resumes...")
    
    matches = []
    
    # In this step, the agent uses its internal "intelligence" (Gemini)
    # to perform the semantic scoring for each professor.
    for prof in profiles:
        # We'll simulate the scoring here, but the Composer Agent
        # will use these scores to select the best resume.
        
        # Simple keyword matching as a fallback/simulation
        scores = {}
        for key, text in resumes.items():
            score = 0
            words = text.lower().split()
            bio = (prof.get("deep_profile_text", "") + " " + prof.get("name", "")).lower()
            for word in words:
                if len(word) > 4 and word in bio:
                    score += 1
            scores[key] = score
        
        best_resume = max(scores, key=scores.get)
        matches.append({
            "name": prof["name"],
            "email": prof.get("email", ""),
            "title": prof.get("title_department", ""),
            "selected_resume": best_resume,
            "alignment_reason": f"Matches key concepts in Resume {best_resume}",
            "profile_link": prof["profile_link"]
        })

    with open("data/matches.json", "w") as f:
        json.dump(matches, f, indent=4)
        
    print(f"Matchmaking complete. Results saved to data/matches.json")
    return matches

if __name__ == "__main__":
    matchmaker()
