"""
Test script for the binary matching API.
"""

import json
import requests
from typing import Dict, Any

def test_binary_api():
    """Test the binary matching API with sample data."""
    
    # Sample candidate (based on your nurse data)
    candidate = {
        "name": "Nurse Candidate",
        "current_title": "Registered Nurse",
        "years_experience": 5,
        "technical_skills": [
            "Patient Care",
            "Case Management", 
            "Healthcare Documentation",
            "Medical-Surgical"
        ],
        "location": "San Francisco",
        "summary": "Experienced RN with case management background"
    }
    
    # Sample job (your home care position)
    job = {
        "job_title": "Home Care Case Manager RN",
        "company": "Home Care Services", 
        "required_skills": [
            "Case Management",
            "Registered Nurse",
            "Patient Care Coordination",
            "Healthcare Documentation",
            "Telehealth",
            "In-home Care"
        ],
        "required_experience_years": 3,
        "location": "Remote/Home-based",
        "remote_allowed": True
    }
    
    # Test data
    test_payload = {
        "candidate": candidate,
        "job": job
    }
    
    try:
        # Make API call
        response = requests.post(
            "http://localhost:5000/evaluate",
            json=test_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response:")
            print(f"   Match: {result['match']}")
            print(f"   Recommendation: {result['recommendation']}")
            print(f"   Binary Result: {result['match']}")  # This is your TRUE/FALSE
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running")
        print("   Run: python api/binary_match_api.py")

def test_health_check():
    """Test the health check endpoint."""
    
    try:
        response = requests.get("http://localhost:5000/health")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health Check:")
            print(f"   Status: {result['status']}")
            print(f"   Model: {result['model']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ API server not running")

if __name__ == "__main__":
    print("Testing Binary Match API")
    print("=" * 50)
    
    print("\n1. Health Check:")
    test_health_check()
    
    print("\n2. Binary Evaluation:")
    test_binary_api()
    
    print("\nDone! 🚀")