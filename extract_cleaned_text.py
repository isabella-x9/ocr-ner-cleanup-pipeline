import json
import os

# Load the JSON OCR data
with open("Vol1StoreOCRPerPage.json", "r") as f:
    ocr_data: dict[str, list[dict]] = json.load(f)

# Target pages to extract
target_pages = {"607", "608", "633", "634", "635"}

# Optional: make output directory
output_dir = "cleaned_outputs"
os.makedirs(output_dir, exist_ok=True)

# Extract and write cleaned text
for page in target_pages:
    entries = ocr_data.get(page, [])
    texts = [entry.get("text", "").strip() for entry in entries if "text" in entry]
    joined_text = "\n".join(texts)

    output_path = os.path.join(output_dir, f"cleaned_page_{page}.txt")
    with open(output_path, "w") as out_file:
        out_file.write(joined_text)
    print(f"✅ Written: {output_path}")
