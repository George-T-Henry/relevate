"""
Generate training data from real user examples in OpenAI eval format.
"""

import json
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from src.schemas import (
    CandidateProfile, JobCriteria, TrainingExample, 
    MatchDecision, ExperienceLevel
)


class RealDataTrainingGenerator:
    """Generate training data from real user examples."""
    
    def __init__(self):
        """Initialize the generator."""
        pass
    
    def load_eval_examples(self, filepath: str) -> List[Dict[str, Any]]:
        """Load examples from OpenAI eval JSON format."""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both single dict and list formats
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Invalid JSON format. Expected dict or list.")
    
    def parse_match_criteria(self, criteria_data: Dict[str, Any]) -> JobCriteria:
        """Parse job criteria from user data."""
        
        return JobCriteria(
            job_title=criteria_data.get("job_title", ""),
            company=criteria_data.get("company", ""),
            department=criteria_data.get("department"),
            required_skills=criteria_data.get("required_skills", []),
            preferred_skills=criteria_data.get("preferred_skills", []),
            required_experience_years=criteria_data.get("required_experience_years"),
            experience_level=self._parse_experience_level(criteria_data.get("experience_level")),
            required_languages=criteria_data.get("required_languages", []),
            preferred_languages=criteria_data.get("preferred_languages", []),
            required_frameworks=criteria_data.get("required_frameworks", []),
            preferred_frameworks=criteria_data.get("preferred_frameworks", []),
            required_tools=criteria_data.get("required_tools", []),
            required_education=criteria_data.get("required_education"),
            preferred_education=criteria_data.get("preferred_education"),
            required_certifications=criteria_data.get("required_certifications", []),
            job_description=criteria_data.get("job_description"),
            responsibilities=criteria_data.get("responsibilities", []),
            company_culture=criteria_data.get("company_culture"),
            location=criteria_data.get("location"),
            remote_allowed=criteria_data.get("remote_allowed", True),
            salary_range=criteria_data.get("salary_range"),
        )
    
    def parse_candidate_profile(self, profile_data: Dict[str, Any]) -> CandidateProfile:
        """Parse candidate profile from user data."""
        
        return CandidateProfile(
            name=profile_data.get("name"),
            email=profile_data.get("email"),
            current_title=profile_data.get("current_title"),
            current_company=profile_data.get("current_company"),
            years_experience=profile_data.get("years_experience"),
            experience_level=self._parse_experience_level(profile_data.get("experience_level")),
            technical_skills=profile_data.get("technical_skills", []),
            soft_skills=profile_data.get("soft_skills", []),
            programming_languages=profile_data.get("programming_languages", []),
            frameworks=profile_data.get("frameworks", []),
            tools=profile_data.get("tools", []),
            work_history=profile_data.get("work_history", []),
            education=profile_data.get("education", []),
            certifications=profile_data.get("certifications", []),
            location=profile_data.get("location"),
            remote_preference=profile_data.get("remote_preference"),
            salary_expectation=profile_data.get("salary_expectation"),
            portfolio_url=profile_data.get("portfolio_url"),
            github_url=profile_data.get("github_url"),
            linkedin_url=profile_data.get("linkedin_url"),
            summary=profile_data.get("summary"),
            raw_profile_data=profile_data.get("raw_profile_data"),
        )
    
    def _parse_experience_level(self, level_str: Optional[str]) -> Optional[ExperienceLevel]:
        """Parse experience level from string."""
        
        if not level_str:
            return None
        
        level_map = {
            "entry": ExperienceLevel.ENTRY,
            "junior": ExperienceLevel.JUNIOR,
            "mid": ExperienceLevel.MID,
            "senior": ExperienceLevel.SENIOR,
            "staff": ExperienceLevel.STAFF,
            "principal": ExperienceLevel.PRINCIPAL,
            "executive": ExperienceLevel.EXECUTIVE,
        }
        
        return level_map.get(level_str.lower())
    
    def _parse_match_decision(self, decision_str: str) -> MatchDecision:
        """Parse match decision from various formats."""
        
        decision_lower = decision_str.lower().strip()
        
        # Handle yes/no format
        if decision_lower in ["yes", "y", "true", "1", "good", "match"]:
            return MatchDecision.STRONG_MATCH
        elif decision_lower in ["no", "n", "false", "0", "bad", "no_match"]:
            return MatchDecision.NO_MATCH
        elif decision_lower in ["maybe", "potential", "consider"]:
            return MatchDecision.POTENTIAL_MATCH
        elif decision_lower in ["weak", "unlikely"]:
            return MatchDecision.WEAK_MATCH
        
        # Handle exact decision names
        decision_map = {
            "strong_match": MatchDecision.STRONG_MATCH,
            "potential_match": MatchDecision.POTENTIAL_MATCH,
            "weak_match": MatchDecision.WEAK_MATCH,
            "no_match": MatchDecision.NO_MATCH,
        }
        
        return decision_map.get(decision_lower, MatchDecision.NO_MATCH)
    
    def convert_to_training_examples(self, eval_data: List[Dict[str, Any]]) -> List[TrainingExample]:
        """Convert eval format data to training examples."""
        
        training_examples = []
        
        for item in eval_data:
            try:
                # Parse job criteria
                job_criteria = self.parse_match_criteria(item["job_criteria"])
                
                # Parse each candidate profile and decision
                for candidate_item in item["candidates"]:
                    profile = self.parse_candidate_profile(candidate_item["profile"])
                    
                    # Parse the human decision
                    human_decision = self._parse_match_decision(candidate_item["decision"])
                    
                    # Parse confidence if provided, otherwise estimate
                    confidence = candidate_item.get("confidence", 0.9 if human_decision == MatchDecision.STRONG_MATCH else 0.7)
                    
                    # Get reasoning if provided
                    reasoning = candidate_item.get("reasoning", f"Human decision: {candidate_item['decision']}")
                    
                    # Create training example
                    example = TrainingExample(
                        profile=profile,
                        job_criteria=job_criteria,
                        human_decision=human_decision,
                        human_confidence=float(confidence),
                        human_reasoning=reasoning,
                        annotator_id=item.get("annotator_id", "human_reviewer"),
                        annotation_timestamp=item.get("timestamp", datetime.now().isoformat()),
                        data_source="real_user_data"
                    )
                    
                    training_examples.append(example)
                    
            except Exception as e:
                print(f"Error processing item: {e}")
                print(f"Item data: {item}")
                continue
        
        return training_examples
    
    def augment_with_synthetic_data(
        self, 
        real_examples: List[TrainingExample], 
        synthetic_count: int = 500
    ) -> List[TrainingExample]:
        """Augment real examples with synthetic data to balance the dataset."""
        
        from scripts.generate_training_data import TrainingDataGenerator
        
        synthetic_generator = TrainingDataGenerator()
        
        # Analyze real data distribution
        real_decisions = [ex.human_decision for ex in real_examples]
        real_distribution = {}
        for decision in MatchDecision:
            real_distribution[decision] = real_decisions.count(decision)
        
        print("Real data distribution:")
        for decision, count in real_distribution.items():
            print(f"  {decision.value}: {count}")
        
        # Generate synthetic data to balance
        synthetic_examples = []
        
        # Calculate how many synthetic examples we need for each decision type
        total_real = len(real_examples)
        target_per_class = max(50, (total_real + synthetic_count) // 4)  # Aim for balanced classes
        
        for decision in MatchDecision:
            real_count = real_distribution[decision]
            needed = max(0, target_per_class - real_count)
            
            if needed > 0:
                print(f"Generating {needed} synthetic examples for {decision.value}")
                
                # Generate until we have enough of this decision type
                attempts = 0
                generated_count = 0
                
                while generated_count < needed and attempts < needed * 10:
                    profile = synthetic_generator.generate_candidate_profile()
                    criteria = synthetic_generator.generate_job_criteria()
                    
                    calc_decision, confidence, reasoning = synthetic_generator.calculate_match_decision(profile, criteria)
                    
                    if calc_decision == decision:
                        example = TrainingExample(
                            profile=profile,
                            job_criteria=criteria,
                            human_decision=decision,
                            human_confidence=confidence,
                            human_reasoning=reasoning,
                            annotator_id="synthetic_generator",
                            annotation_timestamp=datetime.now().isoformat(),
                            data_source="synthetic_augmentation"
                        )
                        synthetic_examples.append(example)
                        generated_count += 1
                    
                    attempts += 1
        
        # Combine real and synthetic data
        all_examples = real_examples + synthetic_examples
        random.shuffle(all_examples)
        
        print(f"\nFinal dataset:")
        print(f"  Real examples: {len(real_examples)}")
        print(f"  Synthetic examples: {len(synthetic_examples)}")
        print(f"  Total examples: {len(all_examples)}")
        
        return all_examples
    
    def save_training_data(self, examples: List[TrainingExample], filepath: str):
        """Save training examples in OpenAI JSONL format."""
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            for example in examples:
                openai_format = example.to_openai_format()
                f.write(json.dumps(openai_format) + '\n')
        
        print(f"Saved {len(examples)} training examples to {filepath}")
    
    def create_example_template(self, output_path: str = "data/raw/example_template.json"):
        """Create an example template file showing the expected format."""
        
        template = {
            "job_criteria": {
                "job_title": "Senior Software Engineer",
                "company": "TechCorp Inc",
                "department": "Engineering",
                "required_skills": ["Python", "JavaScript", "REST API", "SQL"],
                "preferred_skills": ["React", "Docker", "AWS"],
                "required_experience_years": 5,
                "experience_level": "senior",
                "required_languages": ["Python", "JavaScript"],
                "preferred_languages": ["TypeScript"],
                "required_frameworks": ["Django", "React"],
                "required_tools": ["Git", "Docker"],
                "job_description": "We are looking for a senior software engineer...",
                "responsibilities": ["Build scalable web applications", "Mentor junior developers"],
                "location": "San Francisco",
                "remote_allowed": True,
                "salary_range": "$120k-150k"
            },
            "candidates": [
                {
                    "profile": {
                        "name": "John Smith",
                        "current_title": "Software Engineer",
                        "current_company": "StartupXYZ",
                        "years_experience": 6,
                        "experience_level": "senior",
                        "technical_skills": ["Python", "JavaScript", "REST API", "SQL", "Docker"],
                        "programming_languages": ["Python", "JavaScript", "TypeScript"],
                        "frameworks": ["Django", "React", "Express"],
                        "tools": ["Git", "Docker", "AWS"],
                        "education": [{"degree": "BS Computer Science", "school": "University of California"}],
                        "location": "San Francisco",
                        "remote_preference": "hybrid",
                        "summary": "Experienced software engineer with full-stack capabilities..."
                    },
                    "decision": "yes",  # or "no", "maybe", "strong_match", "potential_match", "weak_match", "no_match"
                    "confidence": 0.9,  # optional, 0.0 to 1.0
                    "reasoning": "Strong technical skills match, good experience level, location fit"  # optional
                },
                {
                    "profile": {
                        "name": "Jane Doe",
                        "current_title": "Junior Developer",
                        "current_company": "WebDev Co",
                        "years_experience": 2,
                        "technical_skills": ["HTML", "CSS", "JavaScript"],
                        "programming_languages": ["JavaScript"],
                        "frameworks": ["React"],
                        "location": "Remote"
                    },
                    "decision": "no",
                    "confidence": 0.8,
                    "reasoning": "Not enough experience, missing key backend skills"
                }
            ],
            "annotator_id": "hiring_manager_1",  # optional
            "timestamp": "2024-01-15T10:30:00Z"  # optional
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Example template created at: {output_path}")
        print("Fill this template with your real data and save as multiple JSON files in data/raw/")


def main():
    """Generate training data from real examples."""
    
    generator = RealDataTrainingGenerator()
    
    # Create example template if no real data exists
    raw_data_dir = Path("data/raw")
    if not raw_data_dir.exists() or not any(raw_data_dir.glob("*.json")):
        print("No raw data found. Creating example template...")
        generator.create_example_template()
        print("\nPlease:")
        print("1. Fill the template with your real job criteria and candidate profiles")
        print("2. Save as JSON files in data/raw/ directory")
        print("3. Run this script again")
        return
    
    # Load all JSON files from raw data directory
    all_examples = []
    json_files = list(raw_data_dir.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in data/raw/")
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        try:
            eval_data = generator.load_eval_examples(str(json_file))
            examples = generator.convert_to_training_examples(eval_data)
            all_examples.extend(examples)
            print(f"  Loaded {len(examples)} examples")
        except Exception as e:
            print(f"  Error processing {json_file}: {e}")
    
    if not all_examples:
        print("No valid examples found. Please check your JSON files.")
        return
    
    print(f"\nTotal real examples loaded: {len(all_examples)}")
    
    # Optionally augment with synthetic data
    augment = input("Augment with synthetic data? (y/n): ").lower().strip()
    if augment in ['y', 'yes']:
        synthetic_count = int(input("How many synthetic examples to generate? (default: 500): ") or "500")
        all_examples = generator.augment_with_synthetic_data(all_examples, synthetic_count)
    
    # Split into train/validation/eval sets
    random.shuffle(all_examples)
    
    total = len(all_examples)
    train_size = int(total * 0.7)
    val_size = int(total * 0.2)
    
    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:train_size+val_size]
    eval_examples = all_examples[train_size+val_size:]
    
    # Save datasets
    generator.save_training_data(train_examples, "data/training/train_set.jsonl")
    generator.save_training_data(val_examples, "data/validation/val_set.jsonl")
    generator.save_training_data(eval_examples, "data/eval/eval_set.jsonl")
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")
    print(f"  Evaluation: {len(eval_examples)} examples")
    
    # Print final distribution
    print(f"\nFinal training set distribution:")
    decisions = [ex.human_decision for ex in train_examples]
    for decision in MatchDecision:
        count = decisions.count(decision)
        pct = count / len(train_examples) * 100
        print(f"  {decision.value}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()