from transformers import pipeline

# --- Load once when the module is imported ---
print("Loading BERT (fill-mask)...")
bert_fill = pipeline("fill-mask", model="bert-base-uncased")

print("Loading BART (summarization)...")
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- BERT Fill-Mask Function ---
def run_bert(text):
    """
    Inserts a [MASK] token if not present, and runs BERT fill-mask prediction.
    """
    if "[MASK]" not in text:
        words = text.split()
        if len(words) >= 6:
            words[5] = "[MASK]"  # Replace 6th word
        else:
            words.append("[MASK]")  # If short, just add
        masked = " ".join(words)
    else:
        masked = text

    return bert_fill(masked)

# --- BART Summarization Function ---
def run_bart(text):
    """
    Runs BART summarization on input text (first 1024 tokens recommended).
    """
    return bart_summarizer(text, max_length=100, min_length=25, do_sample=False)
