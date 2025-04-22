import logging
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def analyze_complexity_vs_accuracy(
    complexity_metrics: List[dict],
    accuracy_scores: List[int]
) -> None:
    BASE_DIR = Path(__file__).resolve().parent.parent
    reports = BASE_DIR / 'reports'
    reports.mkdir(parents=True, exist_ok=True)

    logger.info("▶ Starting statistical analysis and plot generation")

    df = pd.DataFrame(complexity_metrics)
    df['accuracy'] = accuracy_scores

    corr = df.corr(numeric_only=True).round(2)
    corr_csv = reports / 'correlation_matrix.csv'
    corr.to_csv(corr_csv)
    logger.info("✔ Saved correlation CSV: %s", corr_csv)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', ax=ax)
    ax.set_title('Correlation Matrix')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Metrics')
    heatmap_png = reports / 'correlation_matrix.png'
    fig.tight_layout()
    fig.savefig(heatmap_png)
    plt.close(fig)
    logger.info("✔ Saved heatmap PNG: %s", heatmap_png)

    for col in df.columns.drop('accuracy'):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x=col, y='accuracy', data=df, scatter_kws={'s': 20}, ax=ax)
        ax.set_title(f'{col} vs Accuracy')
        ax.set_xlabel(col)
        ax.set_ylabel('Accuracy')
        scatter_png = reports / f'{col}_vs_accuracy.png'
        fig.tight_layout()
        fig.savefig(scatter_png)
        plt.close(fig)
        logger.info("✔ Saved scatter PNG: %s", scatter_png)

    combined_csv = reports / 'complexity_vs_accuracy_data.csv'
    df.to_csv(combined_csv, index=False)
    logger.info("✔ Saved combined data CSV: %s", combined_csv)

    logger.info("✔ Statistical analysis COMPLETED.")