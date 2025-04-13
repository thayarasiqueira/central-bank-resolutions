import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from flask import Flask, send_from_directory
import os

server = Flask(__name__)

app = dash.Dash(__name__, server=server)

df = pd.read_csv('reports/complexity_metrics_report.csv')

app.layout = html.Div([
    
    dcc.Graph(
        id='complexity-metrics-scatter',
        figure=px.scatter(df, x='avg_sentence_length', y='lexical_density', 
                          size='syntactic_depth', color='flesch_index',
                          hover_data=['avg_sentence_length', 'lexical_density', 'flesch_index', 'syntactic_depth'],
                          log_x=True, size_max=60)
    ),
    
    dcc.Graph(
        id='complexity-metrics-correlation',
        figure=px.imshow(df.corr(), text_auto=True, title='Correlation Matrix')
    ),
    
    html.Div([
        html.Iframe(src='/static/avg_sentence_length_vs_accuracy.html', style={'width': '100%', 'height': '600px'}),
        html.Iframe(src='/static/lexical_density_vs_accuracy.html', style={'width': '100%', 'height': '600px'}),
        html.Iframe(src='/static/flesch_index_vs_accuracy.html', style={'width': '100%', 'height': '600px'}),
        html.Iframe(src='/static/syntactic_depth_vs_accuracy.html', style={'width': '100%', 'height': '600px'})
    ])
])

@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('reports', path)

if __name__ == '__main__':
    app.run_server(debug=True)