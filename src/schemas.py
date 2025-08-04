"""
Data schemas for Relevate candidate-job matching system.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class MatchDecision(Enum):
    """Possible matching decisions."""
    STRONG_MATCH = "strong_match"        # Definitely interview
    POTENTIAL_MATCH = "potential_match"  # Consider for interview
    WEAK_MATCH = "weak_match"           # Probably pass
    NO_MATCH = "no_match"               # Definitely pass


class ExperienceLevel(Enum):
    """Experience levels."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"


@dataclass
class CandidateProfile:
    """Candidate profile data structure."""
    
    # Basic info
    name: Optional[str] = None
    email: Optional[str] = None
    
    # Experience
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    years_experience: Optional[int] = None
    experience_level: Optional[ExperienceLevel] = None
    
    # Skills and technologies
    technical_skills: List[str] = None
    soft_skills: List[str] = None
    programming_languages: List[str] = None
    frameworks: List[str] = None
    tools: List[str] = None
    
    # Experience details
    work_history: List[Dict[str, Any]] = None  # List of job experiences
    education: List[Dict[str, Any]] = None     # Education history
    certifications: List[str] = None
    
    # Location and preferences
    location: Optional[str] = None
    remote_preference: Optional[str] = None  # "remote", "hybrid", "onsite"
    salary_expectation: Optional[str] = None
    
    # Additional data
    portfolio_url: Optional[str] = None
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    summary: Optional[str] = None
    raw_profile_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.technical_skills is None:
            self.technical_skills = []
        if self.soft_skills is None:
            self.soft_skills = []
        if self.programming_languages is None:
            self.programming_languages = []
        if self.frameworks is None:
            self.frameworks = []
        if self.tools is None:
            self.tools = []
        if self.work_history is None:
            self.work_history = []
        if self.education is None:
            self.education = []
        if self.certifications is None:
            self.certifications = []


@dataclass
class JobCriteria:
    """Job criteria and requirements."""
    
    # Basic job info
    job_title: str
    company: str
    department: Optional[str] = None
    
    # Requirements
    required_skills: List[str] = None
    preferred_skills: List[str] = None
    required_experience_years: Optional[int] = None
    experience_level: Optional[ExperienceLevel] = None
    
    # Technical requirements
    required_languages: List[str] = None
    preferred_languages: List[str] = None
    required_frameworks: List[str] = None
    preferred_frameworks: List[str] = None
    required_tools: List[str] = None
    
    # Education and certifications
    required_education: Optional[str] = None
    preferred_education: Optional[str] = None
    required_certifications: List[str] = None
    
    # Job details
    job_description: Optional[str] = None
    responsibilities: List[str] = None
    company_culture: Optional[str] = None
    
    # Location and logistics
    location: Optional[str] = None
    remote_allowed: bool = True
    salary_range: Optional[str] = None
    
    # Weighting factors (0.0 to 1.0)
    skill_weight: float = 0.4
    experience_weight: float = 0.3
    education_weight: float = 0.1
    cultural_fit_weight: float = 0.2
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.required_skills is None:
            self.required_skills = []
        if self.preferred_skills is None:
            self.preferred_skills = []
        if self.required_languages is None:
            self.required_languages = []
        if self.preferred_languages is None:
            self.preferred_languages = []
        if self.required_frameworks is None:
            self.required_frameworks = []
        if self.preferred_frameworks is None:
            self.preferred_frameworks = []
        if self.required_tools is None:
            self.required_tools = []
        if self.required_certifications is None:
            self.required_certifications = []
        if self.responsibilities is None:
            self.responsibilities = []


@dataclass
class MatchResult:
    """Result of candidate-job matching evaluation."""
    
    # Core results
    decision: MatchDecision
    confidence_score: float  # 0.0 to 1.0
    match_percentage: float  # 0.0 to 100.0
    
    # Detailed breakdown
    skill_match_score: float = 0.0
    experience_match_score: float = 0.0
    education_match_score: float = 0.0
    cultural_fit_score: float = 0.0
    
    # Reasoning
    matching_strengths: List[str] = None
    matching_weaknesses: List[str] = None
    key_concerns: List[str] = None
    recommendation_reason: Optional[str] = None
    
    # Metadata
    model_version: Optional[str] = None
    evaluation_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.matching_strengths is None:
            self.matching_strengths = []
        if self.matching_weaknesses is None:
            self.matching_weaknesses = []
        if self.key_concerns is None:
            self.key_concerns = []


@dataclass
class TrainingExample:
    """Training example for fine-tuning."""
    
    # Input data
    profile: CandidateProfile
    job_criteria: JobCriteria
    
    # Ground truth labels
    human_decision: MatchDecision
    human_confidence: float  # 0.0 to 1.0
    human_reasoning: Optional[str] = None
    
    # Metadata
    annotator_id: Optional[str] = None
    annotation_timestamp: Optional[str] = None
    data_source: Optional[str] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning format (JSONL)."""
        
        # Create system prompt
        system_prompt = """You are an expert recruiter evaluating candidate-job fit. 
        Analyze the candidate profile against job criteria and determine if the match is 
        strong enough for a hiring manager to have an initial conversation.
        
        Respond with a JSON object containing:
        - decision: one of "strong_match", "potential_match", "weak_match", "no_match"
        - confidence: float between 0.0 and 1.0
        - reasoning: brief explanation of your decision"""
        
        # Create user prompt with profile and job data
        user_prompt = f"""
        CANDIDATE PROFILE:
        Name: {self.profile.name or 'Not provided'}
        Current Role: {self.profile.current_title or 'Not provided'} at {self.profile.current_company or 'Not provided'}
        Experience: {self.profile.years_experience or 'Not provided'} years
        Technical Skills: {', '.join(self.profile.technical_skills) if self.profile.technical_skills else 'Not provided'}
        Programming Languages: {', '.join(self.profile.programming_languages) if self.profile.programming_languages else 'Not provided'}
        Education: {self.profile.education or 'Not provided'}
        Location: {self.profile.location or 'Not provided'}
        
        JOB CRITERIA:
        Position: {self.job_criteria.job_title} at {self.job_criteria.company}
        Required Skills: {', '.join(self.job_criteria.required_skills) if self.job_criteria.required_skills else 'Not provided'}
        Required Experience: {self.job_criteria.required_experience_years or 'Not provided'} years
        Required Languages: {', '.join(self.job_criteria.required_languages) if self.job_criteria.required_languages else 'Not provided'}
        Location: {self.job_criteria.location or 'Not provided'}
        Remote Allowed: {self.job_criteria.remote_allowed}
        
        Please evaluate this candidate for this position.
        """
        
        # Expected assistant response
        assistant_response = {
            "decision": self.human_decision.value,
            "confidence": self.human_confidence,
            "reasoning": self.human_reasoning or "Decision based on profile analysis"
        }
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": str(assistant_response)}
            ]
        }