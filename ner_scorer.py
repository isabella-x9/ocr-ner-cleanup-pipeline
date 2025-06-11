import json
import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load gold data
examples = []
with Path("gold.jsonl").open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        text = data["text"]
        entities = data["entities"]

        # Create gold Doc and predicted Doc
        gold_doc = nlp.make_doc(text)
        predicted_doc = nlp(text)

        # Build entity spans from character offsets
        spans = [gold_doc.char_span(start, end, label=label) for start, end, label in entities]
        filtered_spans = [span for span in spans if span is not None]

        if len(filtered_spans) != len(entities):
            skipped = len(entities) - len(filtered_spans)
            print(f"⚠️  Line {i}: Skipped {skipped} misaligned entities")

        gold_doc.ents = filtered_spans

        # Create Example with predicted doc first (as your peer advised)
        example = Example(predicted_doc, gold_doc)
        examples.append(example)

# Evaluate
scorer = Scorer()
scores = scorer.score(examples)

# Output scores
print("\n─" * 60)
print("Named Entity Recognition (NER) Evaluation (over gold.jsonl):")
print(f" - Precision: {scores['ents_p']:.2f}")
print(f" - Recall:    {scores['ents_r']:.2f}")
print(f" - F1 Score:  {scores['ents_f']:.2f}")
print("─" * 60)
