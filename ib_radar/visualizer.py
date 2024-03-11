import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from dash import dash_table
from matplotlib import pyplot as plt

data = {
    'rec_name': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    'trk_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6],
    'class_name': ['x', 'd', 'sd', 'd', 'x', 'x', 'd', 'sd', 'd', 'x', 'x', 'd', 'sd', 'd', 'x', 'd']
}
df_summary = pd.DataFrame(data)

# Constants
ALL_TRK_IDS = df_summary.trk_id.tolist()
ALL_REC_NAMES = df_summary.rec_name.tolist()
ALL_CLASS_NAMES = df_summary.class_name.tolist()


def generate_spectrogram(bsn, dwell_id, channel_id, trk_id, rec_name):
    duration = 5
    sampling_rate = 44100
    frequency_range = (20, 20000)
    num_frequencies = 512

    # Generate a random signal
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.random.uniform(-1, 1, len(t))

    # Create spectrogram
    f, t, Sxx = plt.specgram(signal, NFFT=512, Fs=sampling_rate, noverlap=256, mode='psd')

    # Create Plotly figure
    fig = go.Figure()

    # Add heatmap trace for the spectrogram
    fig.add_trace(go.Heatmap(
        x=t,
        y=f,
        z=10 * np.log10(Sxx),  # Convert to dB for better visualization
        colorscale='Viridis',
        zmin=-40,  # Adjust the color scale range as needed
        zmax=10
    ))

    # Set axis labels and title
    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)'
    )

    return fig


def generate_heatmap_figure(bsn, dwell_id, channel_id, trk_id, rec_name):
    """
    Generate a random heatmap figure.
    """
    bsn = int(bsn * 1000)
    dwell_id = int(dwell_id * 1000)
    channel_id = int(channel_id * 1000)

    np.random.seed(bsn + dwell_id + channel_id + trk_id)

    # Choose a random method for generating heatmap data
    method = np.random.choice(['random', 'sin_wave', 'checkerboard'])

    if method == 'random':
        heatmap_data = np.random.rand(10, 10)
    elif method == 'sin_wave':
        x = np.linspace(0, 2 * np.pi, 10)
        y = np.sin(x)
        heatmap_data = np.outer(y, y)
    elif method == 'checkerboard':
        heatmap_data = np.zeros((10, 10))
        heatmap_data[::2, ::2] = 1
        heatmap_data[1::2, 1::2] = 1

    layout = go.Layout(
        xaxis=dict(title='Doppler'),
        yaxis=dict(title='Range'),
    )
    heatmap_trace = go.Heatmap(z=heatmap_data, colorscale='Viridis')

    fig = go.Figure(data=[heatmap_trace], layout=layout)
    return fig


def load_data(trk_id, rec_name):
    """
    Load dummy data based on track ID and recording name.
    """
    r = 5 * trk_id
    c = 2
    t = np.linspace(0, 4 * np.pi, 30)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = c * t
    vx = np.sin(t)
    vy = np.cos(t)
    vz = np.linspace(0, 2, 30)
    p = np.linspace(0, 2, 30)
    bsn = np.linspace(0, 2, 30)
    dwell_id = np.linspace(0, 2, 30)
    channel_id = np.linspace(0, 2, 30)
    df = pd.DataFrame({'t': t, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'p': p,
                       'rec_name': rec_name, 'bsn': bsn, 'dwell_id': dwell_id, 'channel_id': channel_id})
    return df


def create_dash_app(config):
    """
    Create a Dash web application.
    """
    app = dash.Dash(__name__)
    df_transposed = pd.DataFrame({'Attribute': [None], 'Value': [None]})
    print('Creating Dash app')
    columns = [{'name': col, 'id': col} for col in df_transposed.columns],
    data = df_transposed.to_dict('records')

    app.layout = html.Div([
        html.Div([
            dcc.Dropdown(
                id='rec-name-selector',
                options=[{'label': rec_name, 'value': rec_name} for rec_name in ALL_REC_NAMES],
                value=ALL_REC_NAMES[0],
                placeholder="Select Rec Name"
            ),
            dcc.Dropdown(
                id='trk-id-selector',
                options=[{'label': str(trk_id), 'value': trk_id} for trk_id in ALL_TRK_IDS],
                value=ALL_TRK_IDS[0],
                placeholder="Select Track ID"
            ),
            dcc.Graph(id='3d-scatter'),
        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='column-selector',
                options=[],  # Will be populated dynamically based on the DataFrame columns
                multi=True,
                placeholder="Select Columns"
            ),
            dcc.Graph(id='2d-multigraph'),
        ], style={'width': '49%', 'display': 'inline-block'}),


        html.Div([
            dcc.Graph(id='heatmap'),
            dcc.Graph(id='spectrogram'),
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in df_transposed.columns],
                data=df_transposed.to_dict('records'),
                page_current=0,
                page_size=20,
                style_table={'height': '400px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'center', 'minWidth': '100px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            ),
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginTop': '100px'}),

    ])

    @app.callback(
        Output('rec-name-selector', 'options'),
        [Input('trk-id-selector', 'value')]
    )
    def update_rec_name_selector(trk_id):
        rec_names = df_summary[df_summary['trk_id'] == trk_id]['rec_name'].unique()
        options = [{'label': rec_name, 'value': rec_name} for rec_name in rec_names]
        return options

    # update the trk_id-dropdown options based on the selected rec_name
    @app.callback(
        Output('trk-id-selector', 'options'),
        [Input('rec-name-selector', 'value')]
    )
    def update_trk_id_selector(rec_name):
        trk_ids = df_summary[df_summary['rec_name'] == rec_name]['trk_id'].unique()
        options = [{'label': str(trk_id), 'value': trk_id} for trk_id in trk_ids]
        return options

    # Callbacks
    @app.callback(
        Output('column-selector', 'options'),
        [Input('trk-id-selector', 'value'),
         Input('rec-name-selector', 'value')]
    )
    def update_column_selector(trk_id, rec_name):
        df = load_data(trk_id, rec_name)
        options = [{'label': col, 'value': col} for col in df.columns if col not in config['2d_scatter_defaults']]
        return options

    @app.callback(
        Output('2d-multigraph', 'figure'),
        [Input('3d-scatter', 'clickData'),
         Input('column-selector', 'value'),
         Input('2d-multigraph', 'relayoutData'),
         Input('trk-id-selector', 'value'),
         Input('rec-name-selector', 'value')]
    )
    def update_multigraph(clickData, selected_columns, relayoutData, trk_id, rec_name):
        df = load_data(trk_id, rec_name)
        if selected_columns is None:
            selected_columns = []
        traces = []

        # More efficient to calculate min and max once
        for col in selected_columns:
            polyline_trace = go.Scatter(x=df['t'], y=df[col], mode='lines+markers',
                                        name=f'{col} (Polyline)', line=dict(width=2),
                                        visible=True if col in selected_columns else 'legendonly')
            traces.append(polyline_trace)

        if clickData:
            point_number = clickData['points'][0]['pointNumber']
            for trace in traces:
                if 'marker' in trace and 'color' in trace['marker']:
                    trace['marker']['color'][point_number] = 'red'
                    trace['marker']['size'] = [12 if i == point_number else 8 for i in range(len(df))]

        layout = go.Layout(title='2D Multi Graph', showlegend=True,
                           xaxis=dict(title='Time (t)'),
                           yaxis=dict(title='Value'))

        return {'data': traces, 'layout': layout}

    @app.callback(
        Output('3d-scatter', 'figure'),
        [Input('2d-multigraph', 'clickData'),
         Input('trk-id-selector', 'value'),
         Input('rec-name-selector', 'value')]
    )
    def update_3dscatter(clickData_2d, trk_id, rec_name):
        df = load_data(trk_id, rec_name)
        selected_points = [p['pointNumber'] for p in
                           clickData_2d['points']] if clickData_2d and 'points' in clickData_2d else []

        scatter_trace = go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='markers',
            marker=dict(
                size=4,
                color=['blue' if i in selected_points else 'rgba(211, 211, 211, 0.9)' for i in range(len(df))],
                line=dict(width=1)
            )
        )

        line_trace = go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='lines',
            line=dict(color='grey', width=3),
            visible=True if selected_points else 'legendonly'
        )

        start_point_trace = go.Scatter3d(
            x=[df['x'].iloc[0]],
            y=[df['y'].iloc[0]],
            z=[df['z'].iloc[0]],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Start Point'
        )

        end_point_trace = go.Scatter3d(
            x=[df['x'].iloc[-1]],
            y=[df['y'].iloc[-1]],
            z=[df['z'].iloc[-1]],
            mode='markers',
            marker=dict(size=8, color='orange'),
            name='End Point'
        )

        layout = go.Layout(title='3D Scatter Plot',
                           scene=dict(xaxis=dict(title=config['3d_scatter_defaults']['xaxis']),
                                      yaxis=dict(title=config['3d_scatter_defaults']['yaxis']),
                                      zaxis=dict(title=config['3d_scatter_defaults']['zaxis'])),
                           margin=dict(l=0, r=0, b=0, t=0))

        return {'data': [scatter_trace, line_trace, start_point_trace, end_point_trace], 'layout': layout}

    @app.callback(
        Output('heatmap', 'figure'),
        [Input('2d-multigraph', 'clickData'),
         Input('trk-id-selector', 'value'),
         Input('rec-name-selector', 'value')]
    )
    def update_heatmap_from_2d(clickData_2d, trk_id, rec_name):
        df = load_data(trk_id, rec_name)
        if not clickData_2d or 'points' not in clickData_2d or not clickData_2d['points']:
            return generate_heatmap_figure(0, 0, 0, trk_id, rec_name)

        point_number = clickData_2d['points'][0]['pointNumber']

        # Extracting bsn, dwell_id, and channel_id from the loaded DataFrame
        bsn, dwell_id, channel_id = df.iloc[point_number][['bsn', 'dwell_id', 'channel_id']]
        heatmap_fig = generate_heatmap_figure(bsn, dwell_id, channel_id, trk_id, rec_name)
        return heatmap_fig

    @app.callback(
        Output('spectrogram', 'figure'),
        [Input('2d-multigraph', 'clickData'),
         Input('trk-id-selector', 'value'),
         Input('rec-name-selector', 'value')]
    )
    def update_spectogram(clickData_2d, trk_id, rec_name):
        df = load_data(trk_id, rec_name)
        if not clickData_2d or 'points' not in clickData_2d or not clickData_2d['points']:
            return generate_heatmap_figure(0, 0, 0, trk_id, rec_name)

        point_number = clickData_2d['points'][0]['pointNumber']

        # Extracting bsn, dwell_id, and channel_id from the loaded DataFrame
        bsn, dwell_id, channel_id = df.iloc[point_number][['bsn', 'dwell_id', 'channel_id']]
        heatmap_fig = generate_heatmap_figure(bsn, dwell_id, channel_id, trk_id, rec_name)
        return heatmap_fig

    @app.callback(
        Output('table', 'data'),
        [Input('2d-multigraph', 'clickData'),
         Input('trk-id-selector', 'value'),
         Input('rec-name-selector', 'value')]
    )
    def update_table(clickData_2d, trk_id, rec_name):
        df = load_data(trk_id, rec_name)
        try:
            if not clickData_2d or 'points' not in clickData_2d or not clickData_2d['points']:
                return []  # Return an empty array if there is no data

            point_number = clickData_2d['points'][0]['pointNumber']

            if point_number < 0 or point_number >= len(df):
                return []  # Return an empty array if pointNumber is out of range

            # Extracting relevant data from the loaded DataFrame
            relevant_df = pd.DataFrame(df.iloc[[point_number]])
            df_transposed = relevant_df.T.reset_index()
            df_transposed.columns = ['Attribute', 'Value']
            data = df_transposed.to_dict('records')
            return data

        except Exception as e:
            return []  # Return an empty array in case of an error

    return app


def main():
    """
    Main function to run the Dash application.
    """
    config = {
        '2d_scatter_defaults': [],
        '3d_scatter_defaults': {'xaxis': 'X', 'yaxis': 'Y', 'zaxis': 'Z'}
    }

    app = create_dash_app(config)
    app.run_server(debug=True)


if __name__ == '__main__':
    main()
