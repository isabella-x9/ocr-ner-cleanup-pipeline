import json
import os
from model_utils import run_bert, run_bart
from scorer_wrapper import run_ner_scorer

# === Setup ===
OCR_PATH = "/Users/izzi/Desktop/Duke/Data+/Python/Vol1StoreOCRPerPage.json" 
OUTPUT_DIR = "outputs"
PAGES = [633, 634, 635, 607, 608]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load OCR Text ===
with open(OCR_PATH) as f:
    ocr_data = json.load(f)

# === Run Pipeline for Each Page ===
for page in PAGES:
    pid = str(page)
    text = ocr_data.get(pid, "").strip()

    if not text:
        print(f"[Warning] Page {pid} not found in OCR data.")
        continue

    print(f"\n--- PAGE {pid} ---")

    # Define output file paths up front
    bert_file = os.path.join(OUTPUT_DIR, f"bert_{pid}.txt")
    bart_file = os.path.join(OUTPUT_DIR, f"bart_{pid}.txt")

    # --- BERT: Masked LM ---
    try:
        # Use a short snippet and inject a [MASK] token safely
        short_text = text[:250]
        bert_out = run_bert(short_text)
        bert_out_str = "\n".join([f"{x['sequence']} ({x['score']:.3f})" for x in bert_out])
        with open(bert_file, "w") as f:
            f.write(bert_out_str)
        run_ner_scorer(bert_file)
    except Exception as e:
        print(f"[Error] BERT failed on page {pid}: {e}")

    # --- BART: Summarization ---
    try:
        bart_out = run_bart(text[:1024])  # BART handles longer input
        bart_summary = bart_out[0]["summary_text"]
        with open(bart_file, "w") as f:
            f.write(bart_summary)
        run_ner_scorer(bart_file)
    except Exception as e:
        print(f"[Error] BART failed on page {pid}: {e}")