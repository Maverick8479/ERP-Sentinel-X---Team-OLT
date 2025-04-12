# app.py — Unified Streamlit App (Chatbot + Admin Dashboard)
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import random
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# App setup
st.set_page_config(page_title="ERP AI Sentinel", page_icon="🤖", layout="wide")
st.sidebar.title("🔀 Navigation")
page = st.sidebar.radio("Go to", ["Chatbot 🤖", "Dashboard 📊"])

# Load embedding model and Chroma DB
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Mistral LLM setup (requires MISTRAL_API_KEY as env variable)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0, max_retries=2)

# Response function
@st.cache_resource(show_spinner=False)
def get_response(user_query, chathistory, doc_context):
    docs = db.similarity_search(user_query, k=1)
    combined_context = "\n\nDocuments: " + "\n".join([doc.page_content for doc in docs])

    template = ChatPromptTemplate.from_template("""
        You are an expert ERP support assistant. Use only the context below:
        Context:
        {combined_context}

        Question:
        {user_query}

        Chat history:
        {chathistory}
        Answer:
    """)

    chain = template | llm | StrOutputParser()
    return chain.stream({
        "user_query": user_query,
        "chathistory": chathistory,
        "combined_context": combined_context,
    })

# ------------------ CHATBOT PAGE ------------------
if page == "Chatbot 🤖":
    st.title("ERP Chatbot Assistant 🤖")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am your ERP assistant. How can I help you?")]

    for msg in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
            st.write(msg.content)

    user_query = st.chat_input("Ask your question here...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)

        doc_context = db.similarity_search(user_query)
        response = st.write_stream(get_response(user_query, st.session_state.chat_history, doc_context))
        st.session_state.chat_history.append(AIMessage(content=response))

# ------------------ DASHBOARD PAGE ------------------
elif page == "Dashboard 📊":
    st.title("📊 Admin Dashboard - ERP FAQ Insights")
    conn = sqlite3.connect("IDMS_ERP.db")
    modules_df = pd.read_sql_query("SELECT * FROM Modules", con=conn)
    faqs_df = pd.read_sql_query("SELECT * FROM FAQs", con=conn)

    # Simulate categories if not already in DB
    categories = ['Taxation', 'Billing', 'Compliance', 'Payments', 'Reporting']
    if 'category' not in faqs_df.columns:
        faqs_df['category'] = faqs_df['module_id'].apply(lambda x: random.choice(categories))

    # FAQs by Module
    faq_count = faqs_df.groupby('module_id')['faq_id'].count().reset_index()
    faq_count = faq_count.merge(modules_df, on='module_id')
    fig1 = px.bar(faq_count, x='module_name', y='faq_id', color='module_name', title='Number of FAQs by Module')

    # Category Distribution
    cat_count = faqs_df['category'].value_counts().reset_index()
    cat_count.columns = ['Category', 'Count']
    fig2 = px.pie(cat_count, names='Category', values='Count', title='FAQ Category Distribution')

    # Top FAQs
    top_qs = faqs_df.groupby('question')['times_asked'].sum().reset_index()
    top_qs = top_qs.sort_values(by='times_asked', ascending=False).head(10)
    fig3 = px.bar(top_qs, x='times_asked', y='question', orientation='h', title='Top 10 FAQs')
    fig3.update_layout(yaxis={'autorange': 'reversed'})

    # Avg Ratings
    avg_rating = faqs_df.groupby('module_id')['rating'].mean().reset_index()
    avg_rating = avg_rating.merge(modules_df, on='module_id')
    fig4 = px.bar(avg_rating, x='module_name', y='rating', color='module_name', title='Customer Satisfaction')

    # Solved vs. Unsolved
    if 'solved' in faqs_df.columns:
        solved_data = faqs_df['solved'].value_counts().reset_index()
        solved_data.columns = ['Solved', 'Count']
        fig5 = px.pie(solved_data, values='Count', names='Solved', title='Solved vs Unsolved FAQs')
    else:
        fig5 = None

    # Display Charts
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
    if fig5:
        st.plotly_chart(fig5, use_container_width=True)

    conn.close()
