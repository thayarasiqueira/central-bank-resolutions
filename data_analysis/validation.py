import logging
import random
import json
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_sample(data_path: Union[str, Path], sample_size: float = 0.1) -> None:
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            resolutions = json.load(f)

        total = len(resolutions)
        n = max(1, int(total * sample_size)) if total > 0 else 0
        if n == 0:
            logger.error("❌ No resolutions found in dataset.")
            return

        sample = random.sample(resolutions, n)
        report_path = REPORTS_DIR / 'sample_validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            for i, res in enumerate(sample, start=1):
                report = (
                    f"Sample {i}:\n"
                    f"Title: {res.get('title', 'N/A')}\n"
                    f"Content: {res.get('content', '')[:200]}...\n"
                    f"URL: {res.get('url', 'N/A')}\n"
                    + "-" * 40 + "\n"
                )
                f.write(report)

        logger.info("✔ Sample validation completed: %s", report_path)

    except Exception as e:
        logger.exception("❌ Failed to validate sample: %s", e)