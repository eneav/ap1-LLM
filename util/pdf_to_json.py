import os
import json
from pathlib import Path
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def convert_and_save(input_folder: str, output_folder: str):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".pdf"):
                input_path = os.path.join(root, file)
                raw_text = extract_text_from_pdf(input_path)

                json_data = {
                    "quelle": file,
                    "inhalt_rohtext": raw_text
                }

                output_path = os.path.join(
                    output_folder,
                    file.replace(".pdf", ".json")
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    convert_and_save("data/raw_pdfs", "data/converted_json")
