import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('supermarket_sales.csv')

# Sales per branch analysis
sales_per_branch = df.groupby('Branch').agg({'Total':'sum', 'Invoice ID': 'count'})
sales_per_branch.columns = ['Total Sales', 'Number of Transactions']

# Customer demographics
customer_demographics = df.groupby(['Gender', 'Customer type']).size().unstack()

# Product line analysis
product_line_analysis = df.groupby('Product line').agg({'Total': 'sum', 'gross margin percentage': 'mean'})

# Unit price impact on total sales
unit_price_impact = df[['Unit price', 'Total']].corr()

# Average quantity sold
average_quantity = df['Quantity'].mean()

# Tax contributions by branch
tax_contributions = df.groupby('Branch')['Tax 5%'].sum()

# Time series analysis
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
daily_sales = df.groupby(df['DateTime'].dt.date)['Total'].sum()

# Payment methods distribution
payment_methods = df['Payment'].value_counts()

# Ratings correlation with total sales
ratings_correlation = df[['Rating', 'Total']].corr()

# Gross margin analysis
gross_margin_analysis = df.groupby('Product line')['gross margin percentage'].mean()

# COGS analysis
cogs_analysis = df.groupby('Product line').agg({'cogs': 'sum', 'Total': 'sum'})

# Outlier detection
Q1 = df['Total'].quantile(0.25)
Q3 = df['Total'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Total']<(Q1 - 1.5 * IQR)) | (df['Total'] > (Q3 + 1.5 * IQR))]
