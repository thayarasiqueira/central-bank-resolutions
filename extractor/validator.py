import os

def validate_extraction(processed_dir):
    print("Validating extracted data...")
    txt_files = [f for f in os.listdir(processed_dir) if f.endswith(".txt")]

    for txt_file in txt_files:
        file_path = os.path.join(processed_dir, txt_file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content) < 100:
                print(f"Invalid file (too small): {txt_file}")
            else:
                print(f"File validated successfully: {txt_file}")

    print("Validation completed.")
