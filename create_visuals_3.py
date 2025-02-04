# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:25:34 2025

@author: ZaneC
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:30:42 2025

@author: ZaneC
"""

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

# http://127.0.0.1:8050/

df = pd.read_csv('data/pokemon_dataset.csv')


# Clean and transform some fields
df['generation'] = df['generation'].str.extract(r'(\d+)').astype(int)
df = df.rename(columns={'total_base_stats': 'total'})
df['secondary_type'] = df['secondary_type'].fillna(df['primary_type']).astype('str')

capitalize_fields = ['name', 'primary_type', 'secondary_type', 'category']
for x in capitalize_fields:
    df[x] = df[x].apply(lambda x: x.title())
    
df['first_appearance'] = df['first_appearance'].apply(lambda x: '/'.join([word.title() for word in x.split('/')]))
df['combined_type'] = df['primary_type'] + '/' + df['secondary_type']

# Sort to set order of type_id
df = df.sort_values(by=['primary_type', 'secondary_type'])
type_counts = pd.concat([df['primary_type'], df['secondary_type']]).value_counts()


###################################################

# Calculate average stat by primary_type
stat_categories = ['total','hp','attack','defense','special_attack','special_defense','speed']
df_stats_by_type = df.groupby('primary_type')[stat_categories].mean().round(0).reset_index()
# Add count of rows per primary_type
df_stats_by_type['count'] = df.groupby('primary_type')['primary_type'].transform('size')
for x in stat_categories:
    df_stats_by_type[f'{x}_rank'] = df_stats_by_type.groupby('primary_type')[x].transform('max').rank(ascending=False, method='min')
###
# Calculate average stat by generation
stat_categories = ['total','hp','attack','defense','special_attack','special_defense','speed']
df_stats_by_generation = df.groupby(['first_appearance','generation'])[stat_categories].mean().round(0).reset_index()
for x in stat_categories:
    df_stats_by_generation[f'{x}_rank'] = df_stats_by_generation.groupby(['first_appearance','generation'])[x].transform('max').rank(ascending=False, method='min')
###
# Calculate average stat by pokemon category
stat_categories = ['total','hp','attack','defense','special_attack','special_defense','speed']
df_stats_by_category = df.groupby('category')[stat_categories].mean().round(0).reset_index()
for x in stat_categories:
    df_stats_by_category[f'{x}_rank'] = df_stats_by_category.groupby('category')[x].transform('max').rank(ascending=False, method='min')
### 
# Calculate average stat by generation and category
stat_categories = ['total','hp','attack','defense','special_attack','special_defense','speed']
df_stats_by_generation_category = df.groupby(['first_appearance','generation','category'])[stat_categories].mean().round(0).reset_index()
# Add 'All' as the category
df_stats_by_generation['category'] = 'All'
# Concatenate the two DataFrames
df_stats_by_generation_category = pd.concat([df_stats_by_generation_category, df_stats_by_generation], ignore_index=True)
df_stats_by_generation_category
# Sort for better readability
df_stats_by_generation_category = df_stats_by_generation_category.sort_values(by=['generation', 'category'])
for x in stat_categories:
    df_stats_by_generation_category[f'{x}_rank'] = df_stats_by_generation_category.groupby(['first_appearance','generation','category'])[x].transform('max').rank(ascending=False, method='min')

######################################################
# Initialize Dash app
app = dash.Dash(__name__)



# Set a color map to ensure the colors match between both charts
color_map = {
    "Flying": "#87CEEB",   # Sky Blue
    "Water": "#1E90FF",    # Blue
    "Normal": "#D3D3D3",   # Light Gray
    "Psychic": "#8A2BE2",  # Purple
    "Grass": "#228B22",    # Green
    "Fighting": "#FF0000", # Red
    "Poison": "#800080",   # Purple
    "Fairy": "#FF69B4",    # Pink
    "Ground": "#8B4513",   # Brown
    "Fire": "#FF5733",     # Red
    "Ghost": "#6A5ACD",    # Dark Purple
    "Dragon": "#006400",   # Dark Green
    "Electric": "#FFFF00", # Yellow
    "Steel": "#C0C0C0",    # Silver
    "Dark": "#000000",     # Black
    "Rock": "#A9A9A9",     # Gray
    "Ice": "#ADD8E6",      # Light Blue
    "Bug": "#808000"       # Olive Green
}


donut_chart = go.Figure(data=[go.Pie(
    labels=type_counts.index,
    values=type_counts.values,
    hole=0.3,  # Donut hole size
    name="Type Distribution (Primary & Secondary)",
    marker=dict(colors=[color_map[type_] for type_ in type_counts.index])  # Apply the color map
)])

# Create the stacked bar graph (you have this code already)
def create_stacked_bar(sort_by='total'):
    # Sort by the selected stat (default is 'total')
    df_sorted = df_stats_by_type.sort_values(by=f'{sort_by}_rank', ascending=False)
    
    # Create a stacked bar chart
    fig = px.bar(df_sorted, x='primary_type', y=['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed'],
                 labels={'primary_type': 'Primary Type', 'value': 'Stat Value'},
                 title='Average Stats per Primary Type Across All Generations',
                 color_discrete_map={
                     'hp': '#A7DB8D',
                     'attack': '#FF5959',
                     'defense': '#90A8E0',
                     'special_attack': '#FA92B2',
                     'special_defense': '#C09CF2',
                     'speed': '#F5AC78'
                 })
    
    # Make it a stacked bar chart
    fig.update_traces(
        marker=dict(line=dict(width=0)),  # Remove lines between bar stacks
        texttemplate='%{y}',  # Display values inside the bars
        textposition='inside',  # Position the labels inside the bars
        hoverinfo='y+name'  # Show the value and name when hovering over the bars
    )
    
    # Set chart size
    fig.update_layout(width=1000, height=500, bargap=0.1)
    
    return fig

# Create the line chart
def create_line_chart():
    # Sort by generation
    df_sorted = df_stats_by_generation_category.sort_values(by='generation', ascending=True)
    
    # Create a line chart showing average stats by generation and category
    fig_line = px.line(df_sorted, 
                       x='first_appearance', 
                       y='total',  # Default stat shown, can be changed with a dropdown
                       color='category', 
                       markers=True,
                       labels={'first_appearance': 'Generation', 'total': 'Stat Value'},
                       title='Average Stats by Generation and Category')
    
    # Update layout for better spacing
    fig_line.update_layout(width=750, height=500)
    
    return fig_line


# Create Dash layout
app.layout = html.Div([
    html.Label("Select a stat to sort by:", style={'font-weight': 'bold'}),
    # Dropdown for sorting stats
    dcc.Dropdown(
        id='sort-dropdown',
        options=[
            {'label': 'Total', 'value': 'total'},
            {'label': 'HP', 'value': 'hp'},
            {'label': 'Attack', 'value': 'attack'},
            {'label': 'Defense', 'value': 'defense'},
            {'label': 'Special Attack', 'value': 'special_attack'},
            {'label': 'Special Defense', 'value': 'special_defense'},
            {'label': 'Speed', 'value': 'speed'}
        ],
        value='total',  # Default value
        style={'width': '30%'}
    ),
    
    # Create a div with display: flex to place the graphs side by side
   html.Div([
       # Graph to display the bar chart
       dcc.Graph(
           id='average-stats-graph',
           figure=create_stacked_bar(sort_by='total')  # Initial figure sorted by 'total'
       ),

       # Graph to display the line chart (placed to the right of the bar chart)
       dcc.Graph(
           id='stats-by-generation-graph',
           figure=create_line_chart()
       ),
   ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}),  # Flexbox layout

    # New section for donut charts
    html.Div([
        # Donut chart for Type 1 & Type 2 combined distribution
        dcc.Graph(id="type-donut", 
                  figure=donut_chart, 
                  style={'width': '50%'}),
        
        # Donut chart for the other type (subchart, hidden initially)
        dcc.Graph(id="sub-type-donut", figure={}, style={'width': '37.5%', 'display': 'none'})  # 75% size of the first chart
    ], style={'display': 'flex', 'align-items': 'center'})  # Flexbox to align them horizontally
])

# Callback for the donut chart interactivity
@app.callback(
    Output('sub-type-donut', 'figure'),
    Output('sub-type-donut', 'style'),
    Input('type-donut', 'clickData')
)
def display_subchart(click_data):
    # If no segment is clicked, hide the second donut chart
    if click_data is None:
        return {}, {'display': 'none'}

    # Get the selected type (either from primary_type or secondary_type)
    selected_type = click_data['points'][0]['label']

    # Filter the Pokémon DataFrame where primary_type or secondary_type matches the selected type
    filtered_df = df[(df['primary_type'] == selected_type) | (df['secondary_type'] == selected_type)]

    # Determine the other type for the filtered Pokémon
    # If primary_type is selected, show the distribution of secondary_type
    # If secondary_type is selected, show the distribution of primary_type
    if selected_type in filtered_df['primary_type'].values:
        other_type_counts = filtered_df[filtered_df['primary_type'] == selected_type]['secondary_type'].value_counts()
    else:
        other_type_counts = filtered_df[filtered_df['secondary_type'] == selected_type]['primary_type'].value_counts()

    # Create the sub donut chart for the other type
    sub_donut = go.Figure(data=[go.Pie(
        labels=other_type_counts.index,
        values=other_type_counts.values,
        hole=0.3,  # Donut hole size
        name=f"Other Type Distribution for {selected_type}",
        marker=dict(colors=[color_map[type_] for type_ in other_type_counts.index])  # Apply the same color map
    )])

    # Show the subchart (unhide it)
    return sub_donut, {'display': 'block'}

# Callback to update the graph based on dropdown selection
@app.callback(
    Output('average-stats-graph', 'figure'),
    Input('sort-dropdown', 'value')
)
def update_graph(sort_by):
    # Update the graph when a new stat category is selected
    return create_stacked_bar(sort_by)

if __name__ == '__main__':
    app.run_server(debug=True)