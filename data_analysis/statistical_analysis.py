import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


def analyze_complexity_vs_accuracy(complexity_metrics, accuracy_scores):
    BASE_DIR = Path(__file__).resolve().parent.parent
    reports = BASE_DIR / 'reports'
    reports.mkdir(parents=True, exist_ok=True)

    logger.info("▶ Starting statistical analysis and plot generation")

    df = pd.DataFrame(complexity_metrics)
    df['accuracy'] = accuracy_scores

    corr = df.corr(numeric_only=True).round(2)
    corr_csv = reports / 'correlation_matrix.csv'
    corr.to_csv(corr_csv, index=True)
    logger.info(f"✔ Saved correlation CSV: {corr_csv}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    heatmap_png = reports / 'correlation_matrix.png'
    plt.savefig(heatmap_png)
    plt.close()
    logger.info(f"✔ Saved heatmap PNG: {heatmap_png}")
    
    for col in df.columns[:-1]:
        plt.figure(figsize=(8, 6))
        sns.regplot(x=df[col], y=df['accuracy'], scatter_kws={'s': 20})
        plt.title(f'{col} vs Accuracy')
        plt.xlabel(col)
        plt.ylabel('Accuracy')
        plt.tight_layout()
        scatter_png = reports / f'{col}_vs_accuracy.png'
        plt.savefig(scatter_png)
        plt.close()
        logger.info(f"✔ Saved scatter PNG: {scatter_png}")

    combined_csv = reports / 'complexity_vs_accuracy_data.csv'
    df.to_csv(combined_csv, index=False)
    logger.info(f"✔ Saved combined data CSV: {combined_csv}")

    logger.info("✔ Statistical analysis COMPLETED.")
    