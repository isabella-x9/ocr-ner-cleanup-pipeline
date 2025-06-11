# OCR + NER Cleanup Pipeline

This project processes noisy OCR outputs from historical documents and evaluates Named Entity Recognition (NER) performance using modern NLP models.

## Purpose
We use BERT and BART to "logicalify" (clean up) raw OCR text from the *Records of the Virginia Company* and evaluate how much these transformations improve NER performance using spaCy.

## Project Structure 
```
bert_bart/
├── run_pipeline.py # Main runner script
├── models_utils.py # BERT and BART text processing functions
├── extract_cleaned_text.py # Extract cleaned outputs
├── ner_scorer.py # Scoring script for NER performance
├── gold.jsonl # Ground-truth annotations for evaluation
├── Vol1StoreOCRPerPage.json # Raw OCR input (per-page)
└── outputs/ # Folder where cleaned outputs are stored
```

## How It Works
1. **BERT** (fill-mask): Attempts to reconstruct garbled segments using token prediction.
2. **BART** (summarization): Attempts to rewrite OCR sentences in a logically coherent form.
3. **NER Scorer**: Uses spaCy to evaluate entity extraction quality against annotated `gold.jsonl`.

## Running the Pipeline
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py
