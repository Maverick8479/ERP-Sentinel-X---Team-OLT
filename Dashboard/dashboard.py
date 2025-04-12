import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import sqlite3
import random

conn = sqlite3.connect('IDMS_ERP.db')

modules_df = pd.read_sql_query('SELECT * FROM Modules', con=conn)
faqs_df = pd.read_sql_query('SELECT * FROM FAQs', con=conn)

modules_df = pd.DataFrame({
    'module_id': [1, 2],
    'module_name': ['Accounts & Finance', 'Sales & Dispatch'],
    'description': [
        'Manages GST compliance, tax filing, financial records.',
        'Handles sales invoices, E-way bills, and shipment GST.'
    ]
})

categories = ['Taxation', 'Billing & Invoicing', 'Compliance & Filing', 'Payment & Reconciliation', 'Reporting']
faqs_df['category'] = faqs_df['module_id'].apply(lambda x: random.choice(categories))



faq_count_by_module = faqs_df.groupby('module_id')['faq_id'].count().reset_index()
faq_count_by_module = faq_count_by_module.merge(modules_df, on='module_id')


color_palette = px.colors.sequential.Darkmint

# Bar Chart: FAQs by Module
fig1 = px.bar(faq_count_by_module, x='module_name', y='faq_id', 
              title='Number of FAQs by Module', 
              labels={'faq_id': 'Number of FAQs', 'module_name': 'Module'},
              color='module_name',
              color_discrete_sequence=color_palette)

# Pie Chart: Category Distribution
category_counts = faqs_df['category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']
fig2 = px.pie(category_counts, values='Count', names='Category', 
              title='Distribution of FAQ Categories', 
              color_discrete_sequence=color_palette)

# Bar Chart: Most Frequently Asked Questions
top_questions = faqs_df.groupby('question')['times_asked'].sum().reset_index()
top_questions = top_questions.sort_values(by='times_asked', ascending=False).head(10)
fig3 = px.bar(top_questions, x='times_asked', y='question', 
              orientation='h',
              title='Top 10 Most Frequently Asked Questions',
              labels={'times_asked': 'Number of Times Asked', 'question': 'Question'},
              color_discrete_sequence=['#9AD2B7']) 
fig3.update_layout(yaxis={'autorange': 'reversed'}, coloraxis_showscale=False)

# Bar Chart: Customer Satisfaction
avg_rating_by_module = faqs_df.groupby('module_id')['rating'].mean().reset_index()
avg_rating_by_module = avg_rating_by_module.merge(modules_df, on='module_id')
fig4 = px.bar(avg_rating_by_module, x='module_name', y='rating', 
              title='Customer Satisfaction by Module',
              labels={'rating': 'Average Rating', 'module_name': 'Module'},
              color='module_name',
              color_discrete_sequence=color_palette)

# Pie Chart: Solved vs. Unsolved FAQs
solved_counts = faqs_df['solved'].value_counts().reset_index()
solved_counts.columns = ['Solved', 'Count']
fig5 = px.pie(solved_counts, values='Count', names='Solved', 
              title='Solved vs. Unsolved FAQs', 
              color_discrete_sequence=color_palette)


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='IDMS ERP FAQ Dashboard', style={'textAlign': 'center', 'color': '#0b3d91'}),
    
    html.Div(children='''
        Analytics and insights on FAQs, module performance, and knowledge base gaps.
    ''', style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    dcc.Graph(id='faqs-by-module', figure=fig1),
    dcc.Graph(id='category-distribution', figure=fig2),
    dcc.Graph(id='top-questions', figure=fig3),
    dcc.Graph(id='customer-satisfaction', figure=fig4),
    dcc.Graph(id='solved-unsolved', figure=fig5)
])

if __name__ == '__main__':
    app.run(debug=True)
