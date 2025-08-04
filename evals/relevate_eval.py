"""
Relevate evaluation framework for candidate-job matching models.
Based on OpenAI's evals framework.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import openai
from datetime import datetime

from src.schemas import MatchDecision, TrainingExample, MatchResult


@dataclass
class EvalResult:
    """Results from running an evaluation."""
    
    # Overall metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Per-class metrics
    class_metrics: Dict[str, Dict[str, float]]
    
    # Confusion matrix
    confusion_matrix: List[List[int]]
    
    # Agreement metrics
    strong_match_precision: float  # How often "strong_match" predictions are correct
    hiring_manager_agreement: float  # Agreement with human decisions
    
    # Detailed results
    predictions: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    
    # Metadata
    model_name: str
    eval_timestamp: str
    num_examples: int


class RelevateEvaluator:
    """Evaluator for Relevate candidate matching models."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize evaluator with model name."""
        self.model_name = model_name
        self.client = openai.OpenAI()
    
    def run_eval(self, eval_examples: List[TrainingExample]) -> EvalResult:
        """Run evaluation on a set of examples."""
        
        print(f"Running evaluation on {len(eval_examples)} examples...")
        
        predictions = []
        ground_truth = []
        detailed_results = []
        errors = []
        
        for i, example in enumerate(eval_examples):
            print(f"Evaluating example {i+1}/{len(eval_examples)}")
            
            try:
                # Get model prediction
                prediction = self._get_model_prediction(example)
                predictions.append(prediction["decision"])
                ground_truth.append(example.human_decision.value)
                
                # Store detailed result
                detailed_result = {
                    "example_id": i,
                    "profile_name": example.profile.name,
                    "job_title": example.job_criteria.job_title,
                    "prediction": prediction["decision"],
                    "ground_truth": example.human_decision.value,
                    "confidence": prediction.get("confidence", 0.0),
                    "reasoning": prediction.get("reasoning", ""),
                    "correct": prediction["decision"] == example.human_decision.value
                }
                detailed_results.append(detailed_result)
                
                # Track errors
                if prediction["decision"] != example.human_decision.value:
                    error = {
                        "example_id": i,
                        "predicted": prediction["decision"],
                        "actual": example.human_decision.value,
                        "profile_summary": f"{example.profile.current_title} at {example.profile.current_company}",
                        "job_summary": f"{example.job_criteria.job_title} at {example.job_criteria.company}",
                        "reasoning": prediction.get("reasoning", "")
                    }
                    errors.append(error)
                    
            except Exception as e:
                print(f"Error evaluating example {i}: {e}")
                # Use "no_match" as fallback prediction
                predictions.append("no_match")
                ground_truth.append(example.human_decision.value)
                errors.append({
                    "example_id": i,
                    "error": str(e),
                    "fallback_prediction": "no_match"
                })
        
        # Calculate metrics
        return self._calculate_metrics(
            predictions, 
            ground_truth, 
            detailed_results, 
            errors
        )
    
    def _get_model_prediction(self, example: TrainingExample) -> Dict[str, Any]:
        """Get prediction from the model for a single example."""
        
        # Convert to OpenAI format
        openai_example = example.to_openai_format()
        
        # Get the system and user messages
        messages = openai_example["messages"][:2]  # System + User, no assistant
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Fallback: extract decision from text
                content_lower = content.lower()
                if "strong_match" in content_lower:
                    decision = "strong_match"
                elif "potential_match" in content_lower:
                    decision = "potential_match"
                elif "weak_match" in content_lower:
                    decision = "weak_match"
                else:
                    decision = "no_match"
                
                return {
                    "decision": decision,
                    "confidence": 0.5,
                    "reasoning": content
                }
                
        except Exception as e:
            print(f"Error getting model prediction: {e}")
            return {
                "decision": "no_match",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
    
    def _calculate_metrics(
        self, 
        predictions: List[str], 
        ground_truth: List[str],
        detailed_results: List[Dict[str, Any]],
        errors: List[Dict[str, Any]]
    ) -> EvalResult:
        """Calculate evaluation metrics."""
        
        # Convert to numpy arrays
        pred_array = np.array(predictions)
        truth_array = np.array(ground_truth)
        
        # Overall metrics
        accuracy = accuracy_score(truth_array, pred_array)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth_array, pred_array, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        labels = ["strong_match", "potential_match", "weak_match", "no_match"]
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            truth_array, pred_array, labels=labels, average=None, zero_division=0
        )
        
        class_metrics = {}
        for i, label in enumerate(labels):
            class_metrics[label] = {
                "precision": float(class_precision[i]) if i < len(class_precision) else 0.0,
                "recall": float(class_recall[i]) if i < len(class_recall) else 0.0,
                "f1": float(class_f1[i]) if i < len(class_f1) else 0.0
            }
        
        # Confusion matrix
        cm = confusion_matrix(truth_array, pred_array, labels=labels)
        
        # Special metrics for hiring
        strong_match_mask = pred_array == "strong_match"
        strong_match_correct = (pred_array == truth_array) & strong_match_mask
        strong_match_precision = (
            strong_match_correct.sum() / strong_match_mask.sum() 
            if strong_match_mask.sum() > 0 else 0.0
        )
        
        # Hiring manager agreement (accuracy weighted by importance)
        # Strong matches and no matches are more important to get right
        weights = np.ones(len(predictions))
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if pred in ["strong_match", "no_match"] or truth in ["strong_match", "no_match"]:
                weights[i] = 2.0  # Double weight for critical decisions
        
        weighted_correct = ((pred_array == truth_array) * weights).sum()
        weighted_total = weights.sum()
        hiring_manager_agreement = weighted_correct / weighted_total
        
        return EvalResult(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            class_metrics=class_metrics,
            confusion_matrix=cm.tolist(),
            strong_match_precision=float(strong_match_precision),
            hiring_manager_agreement=float(hiring_manager_agreement),
            predictions=detailed_results,
            errors=errors,
            model_name=self.model_name,
            eval_timestamp=datetime.now().isoformat(),
            num_examples=len(predictions)
        )
    
    def save_eval_results(self, results: EvalResult, filepath: str):
        """Save evaluation results to file."""
        
        # Convert to serializable format
        results_dict = {
            "accuracy": results.accuracy,
            "precision": results.precision,
            "recall": results.recall,
            "f1_score": results.f1_score,
            "class_metrics": results.class_metrics,
            "confusion_matrix": results.confusion_matrix,
            "strong_match_precision": results.strong_match_precision,
            "hiring_manager_agreement": results.hiring_manager_agreement,
            "predictions": results.predictions,
            "errors": results.errors,
            "model_name": results.model_name,
            "eval_timestamp": results.eval_timestamp,
            "num_examples": results.num_examples
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")
    
    def print_summary(self, results: EvalResult):
        """Print a summary of evaluation results."""
        
        print(f"\n{'='*50}")
        print(f"RELEVATE EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Model: {results.model_name}")
        print(f"Examples: {results.num_examples}")
        print(f"Timestamp: {results.eval_timestamp}")
        print()
        
        print(f"OVERALL METRICS:")
        print(f"  Accuracy: {results.accuracy:.3f}")
        print(f"  Precision: {results.precision:.3f}")
        print(f"  Recall: {results.recall:.3f}")
        print(f"  F1 Score: {results.f1_score:.3f}")
        print()
        
        print(f"HIRING-SPECIFIC METRICS:")
        print(f"  Strong Match Precision: {results.strong_match_precision:.3f}")
        print(f"  Hiring Manager Agreement: {results.hiring_manager_agreement:.3f}")
        print()
        
        print(f"PER-CLASS METRICS:")
        for class_name, metrics in results.class_metrics.items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1: {metrics['f1']:.3f}")
        print()
        
        print(f"ERRORS: {len(results.errors)} total")
        if results.errors:
            print("  Top 5 errors:")
            for i, error in enumerate(results.errors[:5]):
                print(f"    {i+1}. Predicted: {error['predicted']}, Actual: {error['actual']}")
                print(f"       {error.get('profile_summary', '')} -> {error.get('job_summary', '')}")
        
        print(f"{'='*50}")


def load_eval_examples(filepath: str) -> List[TrainingExample]:
    """Load evaluation examples from a JSON file."""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        # Convert dict to TrainingExample
        # This would need to be implemented based on your data format
        pass
    
    return examples


if __name__ == "__main__":
    # Example usage
    evaluator = RelevateEvaluator("gpt-4")
    
    # Load eval examples (you'll need to implement this)
    # eval_examples = load_eval_examples("data/eval/test_set.json")
    
    # Run evaluation
    # results = evaluator.run_eval(eval_examples)
    
    # Print and save results
    # evaluator.print_summary(results)
    # evaluator.save_eval_results(results, "evals/results/latest_eval.json")
    
    print("Evaluation framework ready. Add your evaluation data to run tests.")