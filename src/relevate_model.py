"""
Relevate model for candidate-job matching inference.
"""

import json
import openai
from typing import Dict, Any
from src.schemas import CandidateProfile, JobCriteria, MatchResult, MatchDecision


class CandidateMatchingModel:
    """Main interface for Relevate candidate matching."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.client = openai.OpenAI()
    
    def evaluate_match(self, profile: CandidateProfile, criteria: JobCriteria) -> MatchResult:
        """Evaluate candidate-job match."""
        
        system_prompt = """You are an expert recruiter. Analyze the candidate profile against job criteria and return a JSON response with:
        - decision: "strong_match", "potential_match", "weak_match", or "no_match"  
        - confidence: float 0.0-1.0
        - reasoning: brief explanation"""
        
        user_prompt = f"""
        CANDIDATE: {profile.current_title} at {profile.current_company}, {profile.years_experience} years experience
        Skills: {', '.join(profile.technical_skills)}
        Languages: {', '.join(profile.programming_languages)}
        
        JOB: {criteria.job_title} at {criteria.company}
        Required Skills: {', '.join(criteria.required_skills)}
        Required Experience: {criteria.required_experience_years} years
        Required Languages: {', '.join(criteria.required_languages)}
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return MatchResult(
                decision=MatchDecision(result["decision"]),
                confidence_score=result["confidence"],
                match_percentage=result["confidence"] * 100,
                recommendation_reason=result["reasoning"]
            )
        except:
            return MatchResult(
                decision=MatchDecision.NO_MATCH,
                confidence_score=0.5,
                match_percentage=50.0,
                recommendation_reason="Failed to parse model response"
            )