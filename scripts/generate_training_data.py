"""
Generate training data for Relevate candidate-job matching model.
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from src.schemas import (
    CandidateProfile, JobCriteria, TrainingExample, 
    MatchDecision, ExperienceLevel
)


class TrainingDataGenerator:
    """Generate synthetic and curated training data."""
    
    def __init__(self):
        """Initialize with sample data templates."""
        
        # Sample job titles and companies
        self.job_titles = [
            "Senior Software Engineer", "Frontend Developer", "Backend Engineer",
            "Full Stack Developer", "DevOps Engineer", "Data Scientist", 
            "Product Manager", "UX Designer", "Mobile Developer",
            "Machine Learning Engineer", "Site Reliability Engineer",
            "Technical Lead", "Engineering Manager", "Solutions Architect"
        ]
        
        self.companies = [
            "TechCorp", "InnovateCo", "DataDyne", "CloudFirst", "StartupXYZ",
            "Enterprise Solutions", "FinTech Plus", "HealthTech Inc", 
            "EcommerceGiant", "AI Systems", "CyberSec Pro", "DevTools LLC"
        ]
        
        # Skills by category
        self.programming_languages = [
            "Python", "JavaScript", "Java", "TypeScript", "Go", "Rust",
            "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala"
        ]
        
        self.frameworks = [
            "React", "Vue.js", "Angular", "Django", "Flask", "FastAPI",
            "Express", "Spring Boot", "Rails", "Laravel", "Next.js",
            "Svelte", "Node.js", "ASP.NET", ".NET Core"
        ]
        
        self.technical_skills = [
            "REST API", "GraphQL", "Microservices", "Docker", "Kubernetes",
            "AWS", "Azure", "GCP", "CI/CD", "Git", "SQL", "NoSQL",
            "Redis", "PostgreSQL", "MongoDB", "Machine Learning",
            "TensorFlow", "PyTorch", "Apache Kafka", "Elasticsearch"
        ]
        
        self.soft_skills = [
            "Leadership", "Communication", "Problem Solving", "Team Collaboration",
            "Project Management", "Mentoring", "Strategic Thinking",
            "Adaptability", "Time Management", "Analytical Thinking"
        ]
    
    def generate_candidate_profile(self) -> CandidateProfile:
        """Generate a realistic candidate profile."""
        
        experience_years = random.randint(0, 15)
        experience_level = self._get_experience_level(experience_years)
        
        # Select skills based on experience
        num_tech_skills = min(15, max(3, experience_years + random.randint(-2, 5)))
        num_languages = min(8, max(1, (experience_years // 2) + random.randint(0, 3)))
        
        profile = CandidateProfile(
            name=f"Candidate_{random.randint(1000, 9999)}",
            current_title=random.choice(self.job_titles),
            current_company=random.choice(self.companies),
            years_experience=experience_years,
            experience_level=experience_level,
            technical_skills=random.sample(self.technical_skills, num_tech_skills),
            programming_languages=random.sample(self.programming_languages, num_languages),
            frameworks=random.sample(self.frameworks, min(6, max(1, num_languages))),
            soft_skills=random.sample(self.soft_skills, random.randint(3, 6)),
            location=random.choice(["San Francisco", "New York", "Seattle", "Austin", "Remote"]),
            remote_preference=random.choice(["remote", "hybrid", "onsite"]),
        )
        
        return profile
    
    def generate_job_criteria(self) -> JobCriteria:
        """Generate realistic job criteria."""
        
        required_experience = random.randint(2, 10)
        experience_level = self._get_experience_level(required_experience)
        
        num_required_skills = random.randint(3, 8)
        num_preferred_skills = random.randint(2, 6)
        num_required_languages = random.randint(1, 3)
        
        criteria = JobCriteria(
            job_title=random.choice(self.job_titles),
            company=random.choice(self.companies),
            required_skills=random.sample(self.technical_skills, num_required_skills),
            preferred_skills=random.sample(self.technical_skills, num_preferred_skills),
            required_experience_years=required_experience,
            experience_level=experience_level,
            required_languages=random.sample(self.programming_languages, num_required_languages),
            preferred_languages=random.sample(self.programming_languages, random.randint(1, 3)),
            required_frameworks=random.sample(self.frameworks, random.randint(1, 3)),
            location=random.choice(["San Francisco", "New York", "Seattle", "Austin", "Remote"]),
            remote_allowed=random.choice([True, False]),
        )
        
        return criteria
    
    def _get_experience_level(self, years: int) -> ExperienceLevel:
        """Map years of experience to experience level."""
        if years <= 1:
            return ExperienceLevel.ENTRY
        elif years <= 3:
            return ExperienceLevel.JUNIOR
        elif years <= 6:
            return ExperienceLevel.MID
        elif years <= 10:
            return ExperienceLevel.SENIOR
        elif years <= 15:
            return ExperienceLevel.STAFF
        else:
            return ExperienceLevel.PRINCIPAL
    
    def calculate_match_decision(
        self, 
        profile: CandidateProfile, 
        criteria: JobCriteria
    ) -> tuple[MatchDecision, float, str]:
        """Calculate realistic match decision based on profile and criteria."""
        
        score = 0.0
        reasons = []
        
        # Experience matching (30% weight)
        exp_score = 0.0
        if profile.years_experience >= criteria.required_experience_years:
            exp_score = min(1.0, profile.years_experience / (criteria.required_experience_years + 2))
            if profile.years_experience >= criteria.required_experience_years * 1.5:
                reasons.append("Strong experience match")
        else:
            exp_gap = criteria.required_experience_years - profile.years_experience
            exp_score = max(0.0, 1.0 - (exp_gap / criteria.required_experience_years))
            if exp_gap > 2:
                reasons.append("Experience gap concern")
        
        score += exp_score * 0.3
        
        # Skills matching (40% weight)
        required_skills_match = len(set(profile.technical_skills) & set(criteria.required_skills))
        skills_score = required_skills_match / max(1, len(criteria.required_skills))
        
        if skills_score >= 0.8:
            reasons.append("Excellent skills alignment")
        elif skills_score < 0.4:
            reasons.append("Skills gap in key areas")
        
        score += skills_score * 0.4
        
        # Programming languages (20% weight)
        lang_match = len(set(profile.programming_languages) & set(criteria.required_languages))
        lang_score = lang_match / max(1, len(criteria.required_languages))
        
        if lang_score == 1.0:
            reasons.append("Perfect programming language match")
        elif lang_score < 0.5:
            reasons.append("Missing key programming languages")
        
        score += lang_score * 0.2
        
        # Location/remote preference (10% weight)
        location_score = 1.0
        if not criteria.remote_allowed and profile.location != criteria.location:
            location_score = 0.0
            reasons.append("Location mismatch")
        
        score += location_score * 0.1
        
        # Convert score to decision
        reasoning = "; ".join(reasons) if reasons else "Standard profile evaluation"
        
        if score >= 0.85:
            return MatchDecision.STRONG_MATCH, score, f"Strong candidate: {reasoning}"
        elif score >= 0.65:
            return MatchDecision.POTENTIAL_MATCH, score, f"Potential fit: {reasoning}"
        elif score >= 0.35:
            return MatchDecision.WEAK_MATCH, score, f"Weak match: {reasoning}"
        else:
            return MatchDecision.NO_MATCH, score, f"Poor fit: {reasoning}"
    
    def generate_training_examples(self, num_examples: int) -> List[TrainingExample]:
        """Generate a set of training examples."""
        
        examples = []
        
        for i in range(num_examples):
            profile = self.generate_candidate_profile()
            criteria = self.generate_job_criteria()
            
            decision, confidence, reasoning = self.calculate_match_decision(profile, criteria)
            
            example = TrainingExample(
                profile=profile,
                job_criteria=criteria,
                human_decision=decision,
                human_confidence=confidence,
                human_reasoning=reasoning,
                annotator_id="synthetic_generator",
                annotation_timestamp=datetime.now().isoformat(),
                data_source="synthetic"
            )
            
            examples.append(example)
        
        return examples
    
    def save_training_data(self, examples: List[TrainingExample], filepath: str):
        """Save training examples in OpenAI JSONL format."""
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            for example in examples:
                openai_format = example.to_openai_format()
                f.write(json.dumps(openai_format) + '\n')
        
        print(f"Saved {len(examples)} training examples to {filepath}")
    
    def create_balanced_dataset(self, total_examples: int) -> List[TrainingExample]:
        """Create a balanced dataset with good distribution of decisions."""
        
        examples = []
        
        # Target distribution
        distribution = {
            MatchDecision.STRONG_MATCH: 0.15,    # 15% - rare but important
            MatchDecision.POTENTIAL_MATCH: 0.25, # 25% - common consideration
            MatchDecision.WEAK_MATCH: 0.35,      # 35% - most common
            MatchDecision.NO_MATCH: 0.25         # 25% - clear rejections
        }
        
        for decision, ratio in distribution.items():
            target_count = int(total_examples * ratio)
            decision_examples = []
            
            # Generate examples until we have enough of this decision type
            attempts = 0
            while len(decision_examples) < target_count and attempts < target_count * 10:
                profile = self.generate_candidate_profile()
                criteria = self.generate_job_criteria()
                
                calc_decision, confidence, reasoning = self.calculate_match_decision(profile, criteria)
                
                if calc_decision == decision:
                    example = TrainingExample(
                        profile=profile,
                        job_criteria=criteria,
                        human_decision=decision,
                        human_confidence=confidence,
                        human_reasoning=reasoning,
                        annotator_id="synthetic_generator",
                        annotation_timestamp=datetime.now().isoformat(),
                        data_source="synthetic_balanced"
                    )
                    decision_examples.append(example)
                
                attempts += 1
            
            examples.extend(decision_examples)
            print(f"Generated {len(decision_examples)} examples for {decision.value}")
        
        # Shuffle the examples
        random.shuffle(examples)
        return examples


def main():
    """Generate training data for Relevate."""
    
    generator = TrainingDataGenerator()
    
    print("Generating Relevate training data...")
    
    # Generate balanced training set
    train_examples = generator.create_balanced_dataset(1000)
    generator.save_training_data(train_examples, "data/training/train_set.jsonl")
    
    # Generate validation set
    val_examples = generator.create_balanced_dataset(200)
    generator.save_training_data(val_examples, "data/validation/val_set.jsonl")
    
    # Generate evaluation set
    eval_examples = generator.create_balanced_dataset(100)
    generator.save_training_data(eval_examples, "data/eval/eval_set.jsonl")
    
    print("\nTraining data generation complete!")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Evaluation examples: {len(eval_examples)}")
    
    # Print distribution summary
    for dataset_name, examples in [
        ("Training", train_examples),
        ("Validation", val_examples),
        ("Evaluation", eval_examples)
    ]:
        print(f"\n{dataset_name} distribution:")
        decisions = [ex.human_decision for ex in examples]
        for decision in MatchDecision:
            count = decisions.count(decision)
            pct = count / len(examples) * 100
            print(f"  {decision.value}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()