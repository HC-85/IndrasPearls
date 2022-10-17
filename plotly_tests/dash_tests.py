#Import libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
from dash import html

t = np.linspace(0, np.pi*6, 100)
f = np.cos(t)
#Initialize the plot
fig=go.Figure()
#Add the lines
fig.add_trace(go.Scatter(x=t, y=f, visible=True, name='Auto'))
#Add title/set graph size
fig.update_layout(title='Cosine', width=850, height=400)
fig.show()

#Create the app
app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False)