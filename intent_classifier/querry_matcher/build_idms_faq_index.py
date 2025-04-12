import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# FAQs from your PDF (Q1 - Q20)
faq_data = [
    {
        "question": "What is GST and why is it important for businesses?",
        "answer": "GST is an indirect tax levied on the supply of goods and services in India. It replaces multiple indirect taxes and ensures a unified taxation system."
    },
    {
        "question": "How does IDMS help in GST compliance?",
        "answer": "IDMS ERP integrates GST into every transaction, ensuring automatic tax calculations, validation of GSTIN, real-time invoice generation, and GST return filing support."
    },
    {
        "question": "What are the different types of GST in IDMS?",
        "answer": "CGST, SGST, IGST, and UTGST are applied based on whether the transaction is intra-state, inter-state, or within a Union Territory."
    },
    {
        "question": "What is the role of HSN & SAC codes in IDMS?",
        "answer": "HSN codes classify goods and SAC codes classify services for GST. IDMS assigns these for accurate taxation."
    },
    {
        "question": "How does E-Invoicing work in IDMS?",
        "answer": "E-invoices are generated and validated through the Government’s IRP, with a unique IRN and QR code."
    },
    {
        "question": "When is an E-Way Bill required?",
        "answer": "For goods worth more than ₹50,000 being transported, an E-Way Bill must be generated in IDMS with transporter and route details."
    },
    {
        "question": "What is the Reverse Charge Mechanism (RCM) in GST?",
        "answer": "RCM shifts the GST payment liability from supplier to buyer for specific transactions."
    },
    {
        "question": "How does IDMS handle tax invoice vs. proforma invoice?",
        "answer": "A proforma invoice is a preliminary bill, while a tax invoice is legal. IDMS automates conversion."
    },
    {
        "question": "Can IDMS generate GST returns automatically?",
        "answer": "Yes, IDMS compiles data and generates GSTR-1, GSTR-3B, and GSTR-2A reports."
    },
    {
        "question": "How does IDMS help in reconciling GST mismatches?",
        "answer": "It provides mismatch reports before filing returns for accurate tax reconciliation."
    },
    {
        "question": "What is GSTR-1 and how does IDMS help in filing it?",
        "answer": "GSTR-1 details outward supplies. IDMS compiles sales data and generates reports for upload."
    },
    {
        "question": "What is GSTR-3B and how does IDMS assist in its filing?",
        "answer": "GSTR-3B is a monthly summary return. IDMS auto-computes GST liabilities and simplifies filing."
    },
    {
        "question": "How does IDMS handle ITC (Input Tax Credit)?",
        "answer": "IDMS maintains an ITC ledger, matches claims with GSTR-2A, and reconciles discrepancies."
    },
    {
        "question": "What happens if there is a GST mismatch in IDMS?",
        "answer": "IDMS flags mismatches, provides reconciliation reports, and suggests corrections."
    },
    {
        "question": "How does IDMS manage GST for inter-state vs. intra-state transactions?",
        "answer": "It applies CGST/SGST for intra-state, IGST for inter-state, and handles exports and RCM."
    },
    {
        "question": "How are HSN-wise summary reports generated in IDMS?",
        "answer": "IDMS compiles HSN-wise tax summaries in GST-compliant formats."
    },
    {
        "question": "What happens if a sales invoice is cancelled after GST has been filed?",
        "answer": "IDMS issues a Credit Note to adjust tax liabilities in the next cycle."
    },
    {
        "question": "How does IDMS handle GST-exempt and zero-rated supplies?",
        "answer": "It supports GST-exempt goods, zero-rated exports, and GST-exempt customers."
    },
    {
        "question": "How does IDMS automate GST payments?",
        "answer": "IDMS calculates liabilities, generates Challans (PMT-06), and supports online payment modes."
    },
    {
        "question": "How does IDMS generate audit reports for GST compliance?",
        "answer": "IDMS maintains audit trails, tracks invoice changes, and generates GSTR-9 and GSTR-9C."
    }
]

# Extract questions
questions = [faq["question"] for faq in faq_data]

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(questions, convert_to_numpy=True)
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index & metadata
faiss.write_index(index, "idms_faq_index.index")

# Save answers metadata
with open("idms_faq_metadata.pkl", "wb") as f:
    pickle.dump({"questions": questions, "answers": [f["answer"] for f in faq_data]}, f)

print("✅ IDMS ERP FAQ Index Built & Saved!")
