# Relevate

An intelligent candidate-job matching system using OpenAI fine-tuned models to pre-screen candidates for hiring managers.

## Overview

Relevate fine-tunes OpenAI models to evaluate whether a candidate profile matches job criteria strongly enough to warrant an initial conversation with a hiring manager.

**Input**: 
- Candidate profile data
- Job criteria/requirements

**Output**: 
- Match strength score
- Reasoning for the decision
- Recommendation (interview/pass)

## Project Structure

```
relevate/
├── data/
│   ├── raw/          # Original profile and job data
│   ├── processed/    # Cleaned and formatted data
│   ├── training/     # Training datasets (JSONL)
│   ├── validation/   # Validation datasets
│   └── eval/         # Evaluation datasets
├── models/           # Fine-tuned model artifacts
├── evals/            # Evaluation framework and metrics
├── scripts/          # Training and data processing scripts
├── api/              # API for model inference
├── config/           # Configuration files
├── src/              # Core application code
├── docs/             # Documentation
└── tests/            # Test suites
```

## Key Components

### 1. Data Pipeline
- Profile data ingestion and processing
- Job criteria standardization
- Training data generation with human annotations
- Data validation and quality checks

### 2. Fine-tuning Pipeline
- OpenAI fine-tuning orchestration
- Hyperparameter optimization
- Model versioning and management
- Training monitoring and logging

### 3. Evaluation Framework
- Custom evals for hiring relevance
- Metrics: precision, recall, F1, hiring manager agreement
- A/B testing framework
- Performance monitoring

### 4. Inference API
- REST API for real-time predictions
- Batch processing capabilities
- Response caching and optimization
- Integration endpoints for other applications

## Getting Started

1. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate training data:
```bash
python scripts/generate_training_data.py
```

4. Run evaluations:
```bash
python evals/run_eval.py
```

5. Fine-tune model:
```bash
python scripts/fine_tune.py
```

## Usage

```python
from relevate import CandidateMatchingModel

model = CandidateMatchingModel("relevate-v1")
result = model.evaluate_match(profile, job_criteria)
print(f"Match score: {result.score}")
print(f"Recommendation: {result.recommendation}")
```

## Development

More details coming soon...