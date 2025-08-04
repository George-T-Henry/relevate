"""
Binary candidate matching API using structured output.
Returns simple TRUE/FALSE for interview recommendations.
"""

import os
import json
from typing import Dict, Any
from pydantic import BaseModel
import openai
from flask import Flask, request, jsonify

# Pydantic model for structured output
class BinaryMatchResult(BaseModel):
    match: bool  # True = interview, False = reject

class BinaryMatchAPI:
    """Simple binary matching API."""
    
    def __init__(self, model_id: str):
        """Initialize with fine-tuned model ID."""
        self.client = openai.OpenAI()
        self.model_id = model_id
    
    def evaluate_candidate(self, candidate_profile: Dict[str, Any], job_criteria: Dict[str, Any]) -> bool:
        """
        Evaluate candidate and return binary decision.
        
        Returns:
            bool: True = recommend for interview, False = reject
        """
        
        # Format candidate profile
        candidate_text = self._format_candidate(candidate_profile)
        job_text = self._format_job(job_criteria)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert recruiter making binary hiring decisions.
                        
                        Evaluate if this candidate should be interviewed for this position.
                        Consider:
                        - Required skills match
                        - Experience level
                        - Role compatibility
                        
                        Respond with:
                        - true: Recommend for interview (good potential match)
                        - false: Reject (insufficient match)"""
                    },
                    {
                        "role": "user",
                        "content": f"CANDIDATE:\n{candidate_text}\n\nJOB:\n{job_text}\n\nShould this candidate be interviewed?"
                    }
                ],
                response_format=BinaryMatchResult,  # Enforces structured output
                temperature=0.1  # Low temperature for consistent decisions
            )
            
            # Extract boolean result
            return response.choices[0].message.parsed.match
            
        except Exception as e:
            print(f"API Error: {e}")
            return False  # Default to reject on error
    
    def _format_candidate(self, profile: Dict[str, Any]) -> str:
        """Format candidate profile for the model."""
        
        parts = []
        
        if profile.get("name"):
            parts.append(f"Name: {profile['name']}")
        
        if profile.get("current_title"):
            parts.append(f"Current Role: {profile['current_title']}")
        
        if profile.get("years_experience"):
            parts.append(f"Experience: {profile['years_experience']} years")
        
        if profile.get("technical_skills"):
            skills = ", ".join(profile["technical_skills"])
            parts.append(f"Skills: {skills}")
        
        if profile.get("programming_languages"):
            langs = ", ".join(profile["programming_languages"])
            parts.append(f"Languages: {langs}")
        
        if profile.get("location"):
            parts.append(f"Location: {profile['location']}")
        
        if profile.get("summary"):
            parts.append(f"Summary: {profile['summary']}")
        
        return "\n".join(parts)
    
    def _format_job(self, criteria: Dict[str, Any]) -> str:
        """Format job criteria for the model."""
        
        parts = []
        
        if criteria.get("job_title"):
            parts.append(f"Position: {criteria['job_title']}")
        
        if criteria.get("company"):
            parts.append(f"Company: {criteria['company']}")
        
        if criteria.get("required_skills"):
            skills = ", ".join(criteria["required_skills"])
            parts.append(f"Required Skills: {skills}")
        
        if criteria.get("required_experience_years"):
            parts.append(f"Required Experience: {criteria['required_experience_years']} years")
        
        if criteria.get("required_languages"):
            langs = ", ".join(criteria["required_languages"])
            parts.append(f"Required Languages: {langs}")
        
        if criteria.get("location"):
            parts.append(f"Location: {criteria['location']}")
        
        if criteria.get("remote_allowed"):
            parts.append(f"Remote: {'Yes' if criteria['remote_allowed'] else 'No'}")
        
        return "\n".join(parts)


# Flask API
app = Flask(__name__)

# Initialize API (you'll need to update with your actual model ID)
FINE_TUNED_MODEL_ID = os.getenv("RELEVATE_MODEL_ID", "ft:gpt-3.5-turbo:relevate-v1:placeholder")
matcher = BinaryMatchAPI(FINE_TUNED_MODEL_ID)

@app.route("/evaluate", methods=["POST"])
def evaluate_candidate():
    """
    Evaluate a candidate for a job.
    
    Expected JSON:
    {
        "candidate": {
            "name": "John Smith",
            "current_title": "Registered Nurse",
            "years_experience": 5,
            "technical_skills": ["Patient Care", "Case Management"],
            "location": "San Francisco"
        },
        "job": {
            "job_title": "Home Care Case Manager RN",
            "company": "Home Care Services",
            "required_skills": ["Case Management", "Registered Nurse"],
            "required_experience_years": 3,
            "location": "Remote"
        }
    }
    
    Returns:
    {
        "match": true,
        "recommendation": "interview"
    }
    """
    
    try:
        data = request.get_json()
        
        if not data or "candidate" not in data or "job" not in data:
            return jsonify({"error": "Missing candidate or job data"}), 400
        
        # Evaluate candidate
        should_interview = matcher.evaluate_candidate(
            candidate_profile=data["candidate"],
            job_criteria=data["job"]
        )
        
        return jsonify({
            "match": should_interview,
            "recommendation": "interview" if should_interview else "reject"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": FINE_TUNED_MODEL_ID
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)