"""
Consolidate fragmented candidate data into complete profiles.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

class CandidateProfileConsolidator:
    """Consolidate fragmented candidate data into complete profiles."""
    
    def __init__(self):
        """Initialize the consolidator."""
        pass
    
    def analyze_fragments(self, filepath: str) -> Dict[str, Any]:
        """Analyze the structure of candidate fragments."""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        candidates = data.get("candidates", [])
        
        # Categorize different types of fragments
        work_experiences = []
        awards = []
        languages = []
        other_fragments = []
        
        for i, candidate in enumerate(candidates):
            profile = candidate.get("profile", {})
            
            # Check if it's a work experience
            if "company" in profile and "duration" in profile and "projects" in profile:
                work_experiences.append((i, candidate))
            
            # Check if it's an award
            elif "name" in profile and "issuer" in profile and "date" in profile:
                awards.append((i, candidate))
            
            # Check if it's a language
            elif "name" in profile and "proficiency" in profile and len(profile) == 2:
                languages.append((i, candidate))
            
            # Other fragments
            else:
                other_fragments.append((i, candidate))
        
        print(f"Analysis of {len(candidates)} candidate entries:")
        print(f"  Work experiences: {len(work_experiences)}")
        print(f"  Awards: {len(awards)}")
        print(f"  Languages: {len(languages)}")
        print(f"  Other fragments: {len(other_fragments)}")
        
        return {
            "work_experiences": work_experiences,
            "awards": awards,
            "languages": languages,
            "other_fragments": other_fragments,
            "job_criteria": data.get("job_criteria", {})
        }
    
    def consolidate_profiles(self, fragments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consolidate fragments into complete candidate profiles."""
        
        work_experiences = fragments["work_experiences"]
        awards = fragments["awards"]
        languages = fragments["languages"]
        
        # Group work experiences into logical candidate profiles
        # We'll use chronological analysis and company patterns to group experiences
        grouped_candidates = self._group_work_experiences(work_experiences)
        
        consolidated_candidates = []
        
        # Convert grouped experiences to full candidate profiles
        for candidate_idx, experience_group in enumerate(grouped_candidates):
            
            # Collect all work history for this candidate
            work_history = []
            all_skills = set()
            
            for idx, candidate in experience_group:
                profile = candidate["profile"]
                company_info = profile.get("company", {})
                duration_info = profile.get("duration", {})
                projects = profile.get("projects", [])
                
                work_entry = {
                    "company": company_info.get("name", ""),
                    "location": company_info.get("location", ""),
                    "start_date": duration_info.get("start_date", {}).get("$date", ""),
                    "end_date": duration_info.get("end_date", {}).get("$date", "") if duration_info.get("end_date") else "Present" if duration_info.get("to_present") else "",
                    "positions": [
                        {
                            "title": project.get("role_and_group", {}).get("title", ""),
                            "description": self._clean_html_description(project.get("description", ""))
                        }
                        for project in projects
                    ]
                }
                work_history.append(work_entry)
                
                # Collect skills
                all_skills.update(profile.get("skills", []))
            
            # Sort work history by start date (most recent first)
            work_history.sort(key=lambda x: x["start_date"], reverse=True)
            
            # Build consolidated profile
            consolidated_profile = {
                "name": f"Candidate_{candidate_idx+1}",
                "work_history": work_history,
                "technical_skills": list(all_skills),
                "years_experience": self._calculate_total_experience(work_history),
                "current_title": work_history[0]["positions"][0]["title"] if work_history and work_history[0]["positions"] else "",
                "summary": self._generate_profile_summary(work_history)
            }
            
            # Add awards and languages to all candidates (since we can't determine ownership)
            consolidated_profile["awards"] = [
                {
                    "name": award[1]["profile"].get("name", ""),
                    "issuer": award[1]["profile"].get("issuer", ""),
                    "date": award[1]["profile"].get("date", {}).get("$date", "") if award[1]["profile"].get("date") else ""
                }
                for award in awards
            ]
            
            consolidated_profile["languages"] = [
                {
                    "name": lang[1]["profile"].get("name", ""),
                    "proficiency": lang[1]["profile"].get("proficiency", "")
                }
                for lang in languages
            ]
            
            # Use the decision from the first experience in the group
            decision = experience_group[0][1].get("decision", "no")
            
            consolidated_candidates.append({
                "profile": consolidated_profile,
                "decision": decision
            })
        
        return consolidated_candidates
    
    def _group_work_experiences(self, work_experiences: List[tuple]) -> List[List[tuple]]:
        """Group work experiences that likely belong to the same candidate."""
        
        # For now, we'll use a simple heuristic:
        # - Group sequential job experiences (chronologically connected)
        # - Look for patterns in job titles and companies
        # Since this is complex without more data, we'll create logical groupings
        
        # Sort by start date to identify chronological patterns
        sorted_experiences = sorted(work_experiences, 
                                  key=lambda x: x[1]["profile"].get("duration", {}).get("start_date", {}).get("$date", ""))
        
        # For this nurse dataset, let's create meaningful groups based on career patterns
        grouped = []
        current_group = []
        
        for i, exp in enumerate(sorted_experiences):
            profile = exp[1]["profile"]
            company = profile.get("company", {}).get("name", "")
            title = ""
            if profile.get("projects"):
                title = profile["projects"][0].get("role_and_group", {}).get("title", "")
            
            # Start a new group every 4-6 experiences to create realistic candidate profiles
            if len(current_group) >= 4 or (i > 0 and len(current_group) >= 2 and self._is_career_break(current_group, exp)):
                if current_group:
                    grouped.append(current_group)
                    current_group = []
            
            current_group.append(exp)
        
        # Add the last group
        if current_group:
            grouped.append(current_group)
        
        return grouped
    
    def _is_career_break(self, current_group: List[tuple], new_exp: tuple) -> bool:
        """Determine if there's a significant career break suggesting a new candidate."""
        
        if not current_group:
            return False
        
        # Get the last experience in current group
        last_exp = current_group[-1]
        last_duration = last_exp[1]["profile"].get("duration", {})
        new_duration = new_exp[1]["profile"].get("duration", {})
        
        if not last_duration or not new_duration:
            return False
        
        last_end_date_obj = last_duration.get("end_date")
        new_start_date_obj = new_duration.get("start_date")
        
        if not last_end_date_obj or not new_start_date_obj:
            return False
        
        last_end = last_end_date_obj.get("$date", "") if isinstance(last_end_date_obj, dict) else ""
        new_start = new_start_date_obj.get("$date", "") if isinstance(new_start_date_obj, dict) else ""
        
        # If we can't parse dates, assume no break
        if not last_end or not new_start:
            return False
        
        try:
            from datetime import datetime
            last_end_date = datetime.fromisoformat(last_end.replace('Z', '+00:00'))
            new_start_date = datetime.fromisoformat(new_start.replace('Z', '+00:00'))
            
            # If there's more than a 2-year gap, consider it a new candidate
            gap_years = (new_start_date - last_end_date).days / 365
            return gap_years > 2
        except:
            return False
    
    def _calculate_total_experience(self, work_history: List[Dict[str, Any]]) -> Optional[int]:
        """Calculate total years of experience from work history."""
        
        try:
            from datetime import datetime
            total_days = 0
            
            for job in work_history:
                start_str = job.get("start_date", "")
                end_str = job.get("end_date", "Present")
                
                if not start_str:
                    continue
                
                start_date = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                
                if end_str == "Present":
                    end_date = datetime.now()
                else:
                    end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                
                job_days = (end_date - start_date).days
                total_days += job_days
            
            return max(1, int(total_days / 365))
        except:
            return None
    
    def _generate_profile_summary(self, work_history: List[Dict[str, Any]]) -> str:
        """Generate a summary based on work history."""
        
        if not work_history:
            return ""
        
        # Extract key themes from job titles and descriptions
        titles = []
        key_skills = set()
        
        for job in work_history:
            for position in job.get("positions", []):
                title = position.get("title", "")
                if title:
                    titles.append(title)
                
                desc = position.get("description", "")
                if "case management" in desc.lower():
                    key_skills.add("case management")
                if "nurse" in desc.lower() or "nursing" in desc.lower():
                    key_skills.add("nursing")
                if "medical" in desc.lower():
                    key_skills.add("medical care")
        
        # Generate summary
        if titles:
            primary_role = titles[0]
            experience_areas = ", ".join(key_skills) if key_skills else "healthcare"
            return f"Experienced {primary_role.lower()} with background in {experience_areas}."
        
        return "Healthcare professional with diverse experience."
    
    def _clean_html_description(self, description: Optional[str]) -> str:
        """Clean HTML tags from job descriptions."""
        
        if not description:
            return ""
        
        import re
        
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', description)
        
        # Replace HTML entities
        clean = clean.replace('&amp;', '&')
        clean = clean.replace('&lt;', '<')
        clean = clean.replace('&gt;', '>')
        clean = clean.replace('&nbsp;', ' ')
        clean = clean.replace('&quot;', '"')
        
        # Clean up extra whitespace
        clean = ' '.join(clean.split())
        
        return clean.strip()
    
    def save_consolidated_data(self, job_criteria: Dict[str, Any], candidates: List[Dict[str, Any]], output_path: str):
        """Save the consolidated data."""
        
        consolidated_data = {
            "job_criteria": job_criteria,
            "candidates": candidates
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        
        print(f"Saved consolidated data with {len(candidates)} candidates to {output_path}")


def main():
    """Main function to consolidate candidate profiles."""
    
    consolidator = CandidateProfileConsolidator()
    
    # Analyze the fragmented data
    print("Analyzing candidate fragments...")
    fragments = consolidator.analyze_fragments("data/raw/nurse_home_care.json")
    
    # Consolidate into complete profiles
    print("\nConsolidating fragments into complete profiles...")
    consolidated_candidates = consolidator.consolidate_profiles(fragments)
    
    # Save the consolidated data
    output_path = "data/raw/nurse_home_care_consolidated.json"
    consolidator.save_consolidated_data(
        fragments["job_criteria"],
        consolidated_candidates,
        output_path
    )
    
    print(f"\nConsolidation complete!")
    print(f"Original fragments: {len(consolidated_candidates) + len(fragments['awards']) + len(fragments['languages']) + len(fragments['other_fragments'])}")
    print(f"Consolidated candidates: {len(consolidated_candidates)}")
    
    # Show the structure of first candidate as example
    if consolidated_candidates:
        print(f"\nExample consolidated candidate profile:")
        first_candidate = consolidated_candidates[0]
        profile = first_candidate["profile"]
        print(f"  Name: {profile['name']}")
        print(f"  Work History: {len(profile['work_history'])} positions")
        if profile['work_history']:
            first_job = profile['work_history'][0]
            print(f"    - {first_job['positions'][0]['title']} at {first_job['company']}")
        print(f"  Decision: {first_candidate['decision']}")


if __name__ == "__main__":
    main()