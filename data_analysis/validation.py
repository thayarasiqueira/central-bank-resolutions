import logging
import random
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_sample(data_path, sample_size=0.1):
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            resolutions = json.load(f)

        total = len(resolutions)
        n = max(1, int(total * sample_size)) if total > 0 else 0
        if n == 0:
            logger.error("No resolutions found.")
            return

        sample = random.sample(resolutions, n)
        Path('reports').mkdir(exist_ok=True)
        with open('reports/sample_validation_report.txt', 'w', encoding='utf-8') as f:
            for i, res in enumerate(sample, start=1):
                report = (
                    f"Sample {i}:\n"
                    f"Title: {res.get('title', 'N/A')}\n"
                    f"Content: {res.get('content', '')[:200]}...\n"
                    f"URL: {res.get('url', 'N/A')}\n"
                    + "-"*40 + "\n"
                )
                print(report)
                f.write(report)

        logger.info("Sample validation completed.")
    except Exception as e:
        logger.error(f"Error in validate_sample: {e}")
