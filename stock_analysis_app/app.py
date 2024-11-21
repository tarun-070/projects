import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import random

# Load the data
df = pd.read_csv(r"D:\stock_analytic\AAPL.csv")

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Stock Data Dashboard"

# Define the layout of the app
app.layout = html.Div([
    html.H1("Stock Data Dashboard", style={'textAlign': 'center'}),
    
    # Dropdown for selecting date range
    html.Div([
        html.Label("Select Date Range:"),
        dcc.Dropdown(
            id='date-dropdown',
            options=[
                {'label': '5 Days', 'value': '5D'},
                {'label': '1 Month', 'value': '1M'},
                {'label': '6 Months', 'value': '6M'},
                {'label': '1 Year', 'value': '1Y'},
                {'label': 'All', 'value': 'ALL'}
            ],
            value='ALL',
            clearable=False,
            style={'width': '200px'}
        )
    ], style={'marginBottom': '20px'}),

    # Candlestick chart
    dcc.Graph(id='candlestick-chart'),
    
    # Trend graph
    dcc.Graph(id='trend-graph'),
    
    # Random stock purchase result section (from last month)
    html.Div(id='random-purchase-result', style={'marginTop': '30px', 'fontSize': '20px', 'textAlign': 'center'}),

    # Random stock purchase result section (from one year ago)
    html.Div(id='random-purchase-1y-result', style={'marginTop': '30px', 'fontSize': '20px', 'textAlign': 'center'})
])

# Callback to update graphs based on dropdown selection
@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('trend-graph', 'figure'),
     Output('random-purchase-result', 'children'),
     Output('random-purchase-1y-result', 'children')],
    Input('date-dropdown', 'value')
)
def update_graph(selected_range):
    today = df['Date'].max()  # Get the latest date in the data

    # Calculate the start date based on the selected range
    if selected_range == '5D':
        start_date = today - pd.Timedelta(days=5)
    elif selected_range == '1M':
        start_date = today - pd.DateOffset(months=1)
    elif selected_range == '6M':
        start_date = today - pd.DateOffset(months=6)
    elif selected_range == '1Y':
        start_date = today - pd.DateOffset(years=1)
    else:  # 'ALL'
        start_date = df['Date'].min()
    
    # Filter the data to match the selected date range
    mask = (df['Date'] >= start_date) & (df['Date'] <= today)
    filtered_df = df.loc[mask]
    
    # Handle the case when no data is available
    if filtered_df.empty:
        return go.Figure(), go.Figure(), "No data available for the selected range.", "No data available for the random purchase."

    # Create the candlestick chart
    candlestick_fig = go.Figure(data=[go.Candlestick(
        x=filtered_df['Date'],
        open=filtered_df['Open'],
        high=filtered_df['High'],
        low=filtered_df['Low'],
        close=filtered_df['Close']
    )])
    candlestick_fig.update_layout(
        title='Stock Price Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='lightgray',
        xaxis=dict(gridcolor='lightgray', zerolinecolor='lightgray'),
        yaxis=dict(gridcolor='lightgray', zerolinecolor='lightgray'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Create the trend graph
    trend_fig = go.Figure(data=[go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Close'],
        mode='lines+markers',
        name='Close Price'
    )])
    trend_fig.update_layout(
        title='Stock Closing Prices Over Time',
        xaxis_title='Date',
        yaxis_title='Close Price',
        plot_bgcolor='lightgray'
    )
    
    # Simulate a random stock purchase from last month
    # Filter the data for last month
    last_month = today - pd.DateOffset(months=1)
    last_month_data = df[(df['Date'].dt.year == last_month.year) & (df['Date'].dt.month == last_month.month)]
    
    if not last_month_data.empty:
        # Pick a random date from last month's data
        random_date = random.choice(last_month_data['Date'].dt.strftime('%Y-%m-%d').tolist())
        purchase_data = last_month_data[last_month_data['Date'].dt.strftime('%Y-%m-%d') == random_date].iloc[0]
        purchase_price = purchase_data['Close']
        
        # Current price (latest price in the dataset)
        current_price = df[df['Date'] == today].iloc[0]['Close']
        
        # Calculate gain or loss
        gain_loss = current_price - purchase_price
        percentage_change = (gain_loss / purchase_price) * 100
        
        # Format the result message for last month
        result_message_last_month = (
            f"If You bought the stock on a random date 1 month ago on {random_date} at a price of ${purchase_price:.2f}. "
            f"The current price is ${current_price:.2f}. "
            f"Your {('gain' if gain_loss > 0 else 'loss')}: ${gain_loss:.2f} "
            f"({percentage_change:.2f}%)."
        )
    else:
        result_message_last_month = "No data available for a random purchase last month."

    # Simulate a random stock purchase from one year ago
    # Filter the data for one year ago
    one_year_ago = today - pd.DateOffset(years=1)
    one_year_ago_data = df[(df['Date'].dt.year == one_year_ago.year) & (df['Date'].dt.month == one_year_ago.month)]
    
    if not one_year_ago_data.empty:
        # Pick a random date from one year's data
        random_date_1y = random.choice(one_year_ago_data['Date'].dt.strftime('%Y-%m-%d').tolist())
        purchase_data_1y = one_year_ago_data[one_year_ago_data['Date'].dt.strftime('%Y-%m-%d') == random_date_1y].iloc[0]
        purchase_price_1y = purchase_data_1y['Close']
        
        # Current price (latest price in the dataset)
        current_price_1y = df[df['Date'] == today].iloc[0]['Close']
        
        # Calculate gain or loss
        gain_loss_1y = current_price_1y - purchase_price_1y
        percentage_change_1y = (gain_loss_1y / purchase_price_1y) * 100
        
        # Format the result message for one year ago
        result_message_1y = (
            f"If You bought the stock on a random date 1 Year ago on {random_date_1y} at a price of ${purchase_price_1y:.2f}. "
            f"The current price is ${current_price_1y:.2f}. "
            f"Your {('gain' if gain_loss_1y > 0 else 'loss')}: ${gain_loss_1y:.2f} "
            f"({percentage_change_1y:.2f}%)."
        )
    else:
        result_message_1y = "No data available for a random purchase one year ago."

    return candlestick_fig, trend_fig, result_message_last_month, result_message_1y

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=4050)
