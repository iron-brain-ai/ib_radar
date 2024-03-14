import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Global parameters
rec_name = 'rec_name'
track_id = 'track_id'
class_name = 'class_name'
list_of_dets = 'list_of_dets'


def process_data():
    # Sample data
    data = {
        rec_name: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
        track_id: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6],
        class_name: ['z', 'z', 'sd', 'd', 'x', 'x', 'd', 'sd', 'd', 'x', 'x', 'd', 'sd', 'd', 'x', 'd'],
        list_of_dets: [[1, 3, 5], [2, 4], [1, 3, 5], [2, 4], [1, 3, 5], [1, 3, 5], [2, 4], [1, 3, 5], [2, 4], [1, 3, 5],
                       [1, 3, 5], [2, 4], [1, 3, 5], [2, 4], [1, 3, 5], [2, 4]],
    }

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Calculate total counts for each graph
    class_counts = df[class_name].value_counts()
    total_class_counts = pd.concat([class_counts, pd.Series({'Total': class_counts.sum()})])
    rec_counts = df.groupby(rec_name)[track_id].nunique()

    # Add a row for the total count of tracks
    total_rec_counts = pd.concat([rec_counts, pd.Series({'Total': rec_counts.sum()}, name='Total')])

    # Convert record name index to strings (assuming record names are integers)
    total_rec_counts.index = total_rec_counts.index.astype(str)

    # Distribution of the number of detections from each class
    det_counts = df.groupby(class_name)[list_of_dets].apply(lambda x: sum(map(len, x)))
    det_counts = det_counts.reset_index(name='detections_count')

    # Add a row for the total count of detections
    total_detections = det_counts['detections_count'].sum()
    total_detections_df = pd.DataFrame({class_name: ['Total'], 'detections_count': [total_detections]})
    det_counts = pd.concat([det_counts, total_detections_df], ignore_index=True)

    return total_class_counts, det_counts, total_rec_counts


def create_layout(total_class_counts, det_counts, total_rec_counts):
    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Define the layout of the app
    app.layout = html.Div(children=[
        html.Div([
            html.H1("Distribution of Classes and Number of Detections by Class", style={'textAlign': 'center'}),
            dcc.Graph(
                id='class-and-det-distribution',
                figure=go.Figure(data=[
                    # Create a bar chart for the distribution of classes blue and green
                    go.Bar(name='Class Distribution', x=total_class_counts.index, y=total_class_counts),
                    go.Bar(name='Detections by Class', x=det_counts[class_name], y=det_counts['detections_count'])
                ],
                    layout=go.Layout(
                        barmode='group',
                        xaxis=dict(title='Class'),
                        yaxis=dict(title='Count')
                    ))
            )
        ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H1("Distribution of Tracks by Record Names", style={'textAlign': 'center'}),
            dcc.Graph(
                id='tracks-by-rec-names',
                figure=px.bar(total_rec_counts, x=total_rec_counts.index, y=total_rec_counts,
                              labels={'x': 'Record Name', 'y': 'Number of Tracks'}),
            )
        ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ])

    return app


if __name__ == '__main__':
    total_class_counts, det_counts, total_rec_counts = process_data()
    app = create_layout(total_class_counts, det_counts, total_rec_counts)
    app.run_server(debug=True)
