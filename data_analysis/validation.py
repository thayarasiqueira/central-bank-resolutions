import logging
import random
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_sample(data_path, sample_size=0.1):
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            resolutions = json.load(f)
        
        sample = random.sample(resolutions, int(len(resolutions) * sample_size))
        
        with open('reports/sample_validation_report.txt', 'w') as f:
            for i, resolution in enumerate(sample):
                report = (
                    f"Sample {i+1}:\n"
                    f"Title: {resolution['title']}\n"
                    f"Content: {resolution['content'][:200]}...\n"
                    f"URL: {resolution['url']}\n"
                    "-" * 40 + "\n"
                )
                print(report)
                f.write(report)

        logger.info("Sample validation completed successfully.")
    except Exception as e:
        logger.error(f"Error during sample validation: {e}")

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / 'data/raw/resolutions_data.json'
    validate_sample(data_path) 