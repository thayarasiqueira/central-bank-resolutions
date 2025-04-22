import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_trends(data_path):
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
        metrics = df.groupby('year')['content'].apply(lambda x: x.str.len().mean()).reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=metrics, x='year', y='content')
        plt.title('Average Resolution Length Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Characters per Resolution')
        plt.tight_layout()
        trend_png = reports / 'longitudinal_trends.png'
        plt.savefig(trend_png)
        plt.close()
        logger.info(f"✔ Saved longitudinal trend PNG: {trend_png}")

        trend_csv = reports / 'longitudinal_trends_data.csv'
        metrics.to_csv(trend_csv, index=False)
        logger.info(f"✔ Saved longitudinal trend data CSV: {trend_csv}")

        logger.info("✔ Longitudinal analysis COMPLETED.")
    except Exception as e:
        logger.error(f"❌ Failed in longitudinal analysis: {e}")