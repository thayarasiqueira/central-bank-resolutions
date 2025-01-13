import os
from pdfminer.high_level import extract_text

def extract_text_from_pdfs(input_dir, output_dir):
    print("Extracting text from PDFs...")

    for pdf_file in os.listdir(input_dir):
        if pdf_file.endswith(".pdf"):
            input_path = os.path.join(input_dir, pdf_file)
            output_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))

            try:
                text = extract_text(input_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Text extracted to {output_path}.")
            except Exception as e:
                print(f"Error extracting text from {pdf_file}: {e}")

    print("Extraction completed.")
