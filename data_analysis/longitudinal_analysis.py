import logging
import json
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_trends(data_path: Any) -> None:
    BASE_DIR = Path(__file__).resolve().parent.parent
    reports = BASE_DIR / 'reports'
    reports.mkdir(parents=True, exist_ok=True)

    logger.info("▶ Starting longitudinal analysis")

    try:
        text = Path(data_path).read_text(encoding='utf-8')
        resolutions = json.loads(text)

        df = pd.DataFrame(resolutions)
        df['publication_date'] = pd.to_datetime(
            df['publication_date'].str.strip(),
            format='%d/%m/%Y', errors='coerce'
        ).dropna()

        if df.empty:
            logger.error("No valid publication dates.")
            return

        df['year'] = df['publication_date'].dt.year
        metrics = df.groupby('year')['content'].apply(lambda s: s.str.len().mean()).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=metrics, x='year', y='content', ax=ax)
        ax.set_title('Average Resolution Length Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Characters per Resolution')
        trend_png = reports / 'longitudinal_trends.png'
        fig.tight_layout()
        fig.savefig(trend_png)
        plt.close(fig)
        logger.info("✔ Saved longitudinal trend PNG: %s", trend_png)

        trend_csv = reports / 'longitudinal_trends_data.csv'
        metrics.to_csv(trend_csv, index=False)
        logger.info("✔ Saved longitudinal trend data CSV: %s", trend_csv)

        logger.info("✔ Longitudinal analysis COMPLETED.")
    except Exception as e:
        logger.exception("❌ Failed in longitudinal analysis: %s", e)
