import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

logger = logging.getLogger(__name__)

def analyze_complexity_vs_accuracy(complexity_metrics, accuracy_scores):
    try:
        df = pd.DataFrame(complexity_metrics)
        df['accuracy'] = accuracy_scores

        correlation_matrix = df.corr()
        print("Correlation Matrix:")
        print(correlation_matrix)

        fig = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix')
        fig.write_html('reports/correlation_matrix.html')

        for column in df.columns[:-1]:
            fig = px.scatter(df, x=column, y='accuracy', title=f'{column} vs Accuracy')
            fig.write_html(f'reports/{column}_vs_accuracy.html')

        logger.info("Statistical analysis completed successfully.")
        
        with open('reports/statistical_analysis_report.txt', 'w') as f:
            f.write("Statistical Analysis Report\n")
            f.write(correlation_matrix.to_string())
    except Exception as e:
        logger.error(f"Error during statistical analysis: {e}")

if __name__ == "__main__":
    Path('reports').mkdir(parents=True, exist_ok=True)
