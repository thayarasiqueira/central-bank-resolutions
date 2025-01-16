import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def plot_trends(data_path):
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            resolutions = json.load(f)

        df = pd.DataFrame(resolutions)
        
        df['publication_date'] = df['publication_date'].str.strip()
        df['publication_date'] = pd.to_datetime(df['publication_date'], format='%d/%m/%Y', errors='coerce')

        df = df.dropna(subset=['publication_date'])

        if df.empty:
            logger.error("No valid dates found. Please check the date format in the data.")
            return

        df['year'] = df['publication_date'].dt.year
        metrics_over_time = df.groupby('year').agg({
            'content': lambda x: x.str.len().mean(),
        }).reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=metrics_over_time, x='year', y='content')
        plt.title('Average Content Length Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Content Length')
        plt.savefig('reports/longitudinal_trends.png')
        plt.close()

        logger.info("Trends plotted successfully.")
        
        with open('reports/longitudinal_trends_report.txt', 'w') as f:
            f.write("Longitudinal Trends Report\n")
            f.write(metrics_over_time.to_string(index=False))
    except Exception as e:
        logger.error(f"Error plotting trends: {e}")

if __name__ == "__main__":
    Path('reports').mkdir(parents=True, exist_ok=True)
    
    data_path = Path(__file__).resolve().parent.parent / 'data/raw/resolutions_data.json'
    plot_trends(data_path) 