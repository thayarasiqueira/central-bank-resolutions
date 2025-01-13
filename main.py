import os
from data_collection.downloader import download_resolutions
from data_collection.extractor import extract_text_from_pdfs
from data_collection.validator import validate_extraction

def main():
    print("Starting data collection for Central Bank resolutions...")

    # Directories
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Download PDFs
    download_resolutions(raw_dir)

    # Extract text from PDFs
    extract_text_from_pdfs(raw_dir, processed_dir)

    # Validate extracted data
    validate_extraction(processed_dir)

    print("Process completed successfully!")

if __name__ == "__main__":
    main()
