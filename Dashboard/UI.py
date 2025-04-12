import streamlit as st
import sqlite3
import pandas as pd
import subprocess

conn = sqlite3.connect('IDMS_ERP.db')
cursor = conn.cursor()

modules = {
    1: 'Accounts & Finance',
    2: 'Sales & Dispatch'
}

questions = {
    1: [
        "What is GST compliance?",
        "How to file GSTR-1?",
        "What are HSN/SAC codes?",
        "How to claim ITC?",
        "How to handle GST mismatch reconciliation?",
        "What is GSTR-3B?",
        "How to generate audit reports?",
        "How to track transactions?",
        "What is a tax invoice?",
        "How to create financial records?"
    ],
    2: [
        "How to create sales invoices?",
        "What is an E-way bill?",
        "How to handle shipment GST?",
        "How to generate proforma invoices?",
        "How to handle sales tax compliance?",
        "What is the process for dispatch entry?",
        "How to track shipment status?",
        "What are the common errors in dispatch billing?",
        "How to handle GST on exports?",
        "How to generate dispatch reports?"
    ]
}

st.title('Ask Your Question')

module_name = st.selectbox('Select a Module', options=list(modules.values()))
module = list(modules.keys())[list(modules.values()).index(module_name)]

if module:
    question = st.selectbox('Select a Question', options=questions[module])
    rating = st.slider('Rate the Quality of the Answer (1-5)', 1, 5, 3)

    if st.button('Submit'):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO FAQs (module_id, question, rating) VALUES (?, ?, ?)
        ''', (module, question, rating))
        conn.commit()
        st.success('Your question has been recorded!')

conn.close()
