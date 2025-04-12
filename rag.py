import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# App config
st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.title("ERP ChatBot")

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

# Load embedding model and database
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedded_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

def get_response(user_query, chathistory, doc_context):
    docs = embedded_db.similarity_search(user_query, k=1)
    combined_context = "\n\nDocuments: " + "\n".join([doc.page_content for doc in docs])

    llm_query_res_template = """
    You are an expert Customer Relationship Manager with a deep understanding of customer interactions. 
    You provide answers based strictly on the given context, ensuring accuracy and relevance.

    Guidelines for responses:
    - Provide a **detailed explanation** with structured formatting.
    - Ensure responses are **neat, clean, and well-indented** for readability.
    - Keep answers **precise and relevant**, avoiding unnecessary details.
    - Maintain a **professional and friendly** tone in all responses.
    - If applicable, include **relevant news articles or links** related to the topic for further reference.

    Context:
    {combined_context}

    Question:
    {user_query}

    Chat history:
    {chathistory}

    Answer:

    [If available, provide relevant news links or articles.]
"""



    prompt_query_res_template = ChatPromptTemplate.from_template(llm_query_res_template)
    llm_chain = prompt_query_res_template | llm | StrOutputParser()

    return llm_chain.stream({
        "user_query": user_query,
        "chathistory": chathistory,
        "combined_context": combined_context,
    })

# Chatbot session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am from ERP. How can I help you?"),
    ]

# Conversation handling
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# Input Handling
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Perform document similarity search
    doc_context = embedded_db.similarity_search(user_query)

    # Get response from LLM
    response = st.write_stream(get_response(user_query, st.session_state.chat_history, doc_context))
    st.session_state.chat_history.append(AIMessage(content=response))
