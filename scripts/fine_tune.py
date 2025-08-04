"""
Fine-tuning pipeline for Relevate candidate-job matching model.
"""

import os
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import openai


class RelevateFineTuner:
    """Fine-tuning pipeline for Relevate models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize fine-tuner with OpenAI API key."""
        
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("OpenAI API key not provided")
        
        self.client = openai.OpenAI()
    
    def upload_training_file(self, filepath: str) -> str:
        """Upload training data file to OpenAI."""
        
        print(f"Uploading training file: {filepath}")
        
        with open(filepath, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        print(f"Training file uploaded with ID: {file_id}")
        
        return file_id
    
    def upload_validation_file(self, filepath: str) -> Optional[str]:
        """Upload validation data file to OpenAI."""
        
        if not os.path.exists(filepath):
            print("No validation file found, skipping...")
            return None
        
        print(f"Uploading validation file: {filepath}")
        
        with open(filepath, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        print(f"Validation file uploaded with ID: {file_id}")
        
        return file_id
    
    def create_fine_tune_job(
        self,
        training_file_id: str,
        validation_file_id: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a fine-tuning job."""
        
        print(f"Creating fine-tune job with model: {model}")
        
        # Default hyperparameters
        default_hyperparams = {
            "n_epochs": 3,
            "batch_size": 1,
            "learning_rate_multiplier": 2.0
        }
        
        if hyperparameters:
            default_hyperparams.update(hyperparameters)
        
        # Create job parameters
        job_params = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": default_hyperparams
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        if suffix:
            job_params["suffix"] = suffix
        
        # Create the fine-tuning job
        response = self.client.fine_tuning.jobs.create(**job_params)
        
        job_id = response.id
        print(f"Fine-tuning job created with ID: {job_id}")
        
        return job_id
    
    def monitor_fine_tune_job(self, job_id: str, check_interval: int = 60) -> Dict[str, Any]:
        """Monitor fine-tuning job progress."""
        
        print(f"Monitoring fine-tuning job: {job_id}")
        print(f"Check interval: {check_interval} seconds")
        
        while True:
            # Get job status
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")
            
            if status in ["succeeded", "failed", "cancelled"]:
                break
            
            # Print training progress if available
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"  Trained tokens: {job.trained_tokens}")
            
            time.sleep(check_interval)
        
        # Get final job details
        final_job = self.client.fine_tuning.jobs.retrieve(job_id)
        
        result = {
            "job_id": job_id,
            "status": final_job.status,
            "model_id": final_job.fine_tuned_model,
            "created_at": final_job.created_at,
            "finished_at": final_job.finished_at,
            "training_file": final_job.training_file,
            "validation_file": final_job.validation_file,
            "hyperparameters": final_job.hyperparameters,
            "trained_tokens": final_job.trained_tokens,
            "error": final_job.error if final_job.status == "failed" else None
        }
        
        return result
    
    def run_complete_pipeline(
        self,
        training_file_path: str,
        validation_file_path: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        model_suffix: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the complete fine-tuning pipeline."""
        
        print("Starting Relevate fine-tuning pipeline...")
        print(f"Training file: {training_file_path}")
        print(f"Validation file: {validation_file_path}")
        print(f"Base model: {model_name}")
        
        try:
            # Step 1: Upload training data
            training_file_id = self.upload_training_file(training_file_path)
            
            # Step 2: Upload validation data (optional)
            validation_file_id = None
            if validation_file_path:
                validation_file_id = self.upload_validation_file(validation_file_path)
            
            # Step 3: Create fine-tuning job
            job_id = self.create_fine_tune_job(
                training_file_id=training_file_id,
                validation_file_id=validation_file_id,
                model=model_name,
                suffix=model_suffix,
                hyperparameters=hyperparameters
            )
            
            # Step 4: Monitor progress
            result = self.monitor_fine_tune_job(job_id)
            
            # Step 5: Save results
            self.save_fine_tune_results(result)
            
            if result["status"] == "succeeded":
                print(f"\n🎉 Fine-tuning completed successfully!")
                print(f"Model ID: {result['model_id']}")
                return result
            else:
                print(f"\n❌ Fine-tuning failed with status: {result['status']}")
                if result["error"]:
                    print(f"Error: {result['error']}")
                return result
        
        except Exception as e:
            print(f"\n💥 Pipeline failed with error: {str(e)}")
            raise
    
    def save_fine_tune_results(self, result: Dict[str, Any]):
        """Save fine-tuning results to file."""
        
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = models_dir / f"fine_tune_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        
        # If successful, also save a simple model registry entry
        if result["status"] == "succeeded":
            registry_file = models_dir / "model_registry.json"
            
            # Load existing registry or create new
            registry = []
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
            
            # Add new model entry
            model_entry = {
                "model_id": result["model_id"],
                "base_model": "gpt-3.5-turbo",  # Could be extracted from job
                "created_at": result["finished_at"],
                "job_id": result["job_id"],
                "trained_tokens": result["trained_tokens"],
                "status": "active",
                "description": "Relevate candidate-job matching model"
            }
            registry.append(model_entry)
            
            # Save updated registry
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
            
            print(f"Model registered in: {registry_file}")
    
    def list_fine_tune_jobs(self, limit: int = 10) -> list:
        """List recent fine-tuning jobs."""
        
        jobs = self.client.fine_tuning.jobs.list(limit=limit)
        
        print(f"Recent fine-tuning jobs (limit: {limit}):")
        print("-" * 80)
        
        for job in jobs.data:
            print(f"ID: {job.id}")
            print(f"Status: {job.status}")
            print(f"Model: {job.fine_tuned_model or 'N/A'}")
            print(f"Created: {job.created_at}")
            print("-" * 80)
        
        return jobs.data
    
    def delete_fine_tuned_model(self, model_id: str) -> bool:
        """Delete a fine-tuned model."""
        
        try:
            response = self.client.models.delete(model_id)
            print(f"Model {model_id} deletion status: {response.deleted}")
            return response.deleted
        except Exception as e:
            print(f"Error deleting model {model_id}: {str(e)}")
            return False


def main():
    """Main fine-tuning script."""
    
    # Configuration
    training_file = "data/training/train_set.jsonl"
    validation_file = "data/validation/val_set.jsonl"
    
    # Check if training data exists
    if not os.path.exists(training_file):
        print(f"Training file not found: {training_file}")
        print("Please run 'python scripts/generate_training_data.py' first")
        return
    
    # Initialize fine-tuner
    fine_tuner = RelevateFineTuner()
    
    # Custom hyperparameters
    hyperparams = {
        "n_epochs": 3,
        "batch_size": 1,
        "learning_rate_multiplier": 2.0
    }
    
    # Run fine-tuning pipeline
    result = fine_tuner.run_complete_pipeline(
        training_file_path=training_file,
        validation_file_path=validation_file if os.path.exists(validation_file) else None,
        model_name="gpt-3.5-turbo",
        model_suffix="relevate-v1",
        hyperparameters=hyperparams
    )
    
    print("\nFine-tuning pipeline completed!")
    print(f"Final status: {result['status']}")
    
    if result["status"] == "succeeded":
        print(f"\n🚀 Your new model is ready: {result['model_id']}")
        print("You can now use this model for candidate-job matching!")
    

if __name__ == "__main__":
    main()