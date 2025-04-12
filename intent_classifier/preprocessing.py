import spacy
from spacy.matcher import PhraseMatcher
from typing import List, Dict

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define stopwords (you can customize this list)
from spacy.lang.en.stop_words import STOP_WORDS

# Custom synonym/token mapper
SYNONYM_MAP = {
    # 🔹 Common Transaction Synonyms
    "bill": "invoice",
    "invoice copy": "invoice",
    "tax invoice": "invoice",
    "gst invoice": "invoice",
    "final bill": "invoice",
    "payment receipt": "invoice",
    "invoice statement": "invoice",

    "sales order": "so",
    "so": "sales_order",
    "customer order": "sales_order",
    "order confirmation": "order_confirmation_report",

    "purchase order": "po",
    "po": "purchase_order",
    "buying order": "purchase_order",
    "procurement order": "purchase_order",

    "dispatch note": "drn",
    "delivery note": "drn",
    "shipping note": "drn",
    "asn": "asn",
    "shipment alert": "asn",
    "shipment notice": "asn",

    "grn": "goods_receipt_note",
    "goods received note": "goods_receipt_note",
    "material receipt": "goods_receipt_note",
    "goods inward": "goods_receipt_note",
    "grn status": "check_grn_status",

    "gin": "goods_issue_note",
    "goods issued note": "goods_issue_note",
    "material issue": "goods_issue_note",

    "gtr": "goods_transfer_request",
    "stock transfer": "goods_transfer_request",
    "material movement": "goods_transfer_request",

    "batch card": "batch_card",
    "production batch": "batch_card",

    "job card": "job_card",
    "production order": "job_card",
    "work order": "job_card",

    "credit note": "credit_note",
    "debit note": "debit_note",
    "sales credit note": "credit_note",
    "sales debit note": "debit_note",

    # 🔹 Quality/Inspection
    "qc": "quality_control",
    "inspection checklist": "inspection_checklist",
    "material inspection": "material_inspection",
    "pdir": "pre_dispatch_inspection_report",

    # 🔹 Reports
    "stock report": "inventory_report",
    "stock level report": "inventory_report",
    "stock aging report": "stock_aging_report",
    "inventory report": "inventory_report",

    "sales register": "sales_register",
    "po status": "po_status_report",
    "outstanding po": "outstanding_po_report",
    "purchase register": "purchase_register",

    "dispatch status": "dispatch_status_report",
    "inventory levels": "inventory_report",
    "production inventory": "production_inventory_report",
    "finished goods tracking": "finished_goods_report",

    "inspection report": "inspection_report",
    "material revalidation": "material_revalidation",

    # 🔹 Modules
    "sales": "sales_module",
    "purchase": "purchase_module",
    "stores": "stores_module",
    "inventory": "stores_module",
    "stock": "stores_module",
    "production": "production_module",
    "manufacturing": "production_module",
    "quality": "quality_module",
    "qc": "quality_module",
    "dispatch": "dispatch_logistics_module",
    "logistics": "dispatch_logistics_module",
    "shipping": "dispatch_logistics_module",

    # 🔹 Master Data Synonyms
    "sku": "sku_master",
    "product catalog": "sku_master",
    "item master": "item_master",
    "supplier master": "supplier_master",
    "vendor list": "supplier_master",
    "payment terms": "payment_terms",
    "credit terms": "payment_terms",
    "customer master": "b2b_customers",
    "logistics partners": "logistics",

    # 🔹 GST Related
    "e-invoice": "einvoice",
    "einvoice": "einvoice",
    "e way bill": "e_way_bill",
    "ewaybill": "e_way_bill",
    "eway": "e_way_bill",
    "gst return": "gst_report",
    "gstr1": "gst_report",
    "gstr3b": "gst_report",
    "gst summary": "gst_report",
    "itc": "input_tax_credit",
    "input tax": "input_tax_credit",
    "gst mismatch": "gst_reconciliation",
    "gst ledger": "gst_ledger",
    "gst payment challan": "gst_challan",
    "pmt-06": "gst_challan",

    # 🔹 Others
    "challan": "delivery_challan",
    "manual invoice upload": "manual_invoice_upload",
    "credit approval": "finance_approval",
    "proforma": "proforma_invoice",
    "proforma bill": "proforma_invoice",
    "advance shipping notice": "asn"
}



# Phrases for custom matching (NER-style)
ERP_TERMS = list(SYNONYM_MAP.keys())

# Phrase matcher setup for ERP terms
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(term) for term in ERP_TERMS]
matcher.add("ERP_TERMS", patterns)


def normalize_synonyms(text: str) -> str:
    """Replace known phrases with standardized tokens."""
    doc = nlp(text)
    matches = matcher(doc)
    replaced_text = text.lower()
    for match_id, start, end in matches:
        span_text = doc[start:end].text.lower()
        if span_text in SYNONYM_MAP:
            replaced_text = replaced_text.replace(span_text, SYNONYM_MAP[span_text])
    return replaced_text


def preprocess_query(query: str) -> Dict:
    """Complete preprocessing pipeline."""
    # Lowercase and synonym normalization
    normalized = normalize_synonyms(query)
    
    # Tokenization + NLP parsing
    doc = nlp(normalized)

    # Token features
    tokens = []
    lemmas = []
    entities = []

    for token in doc:
        if token.is_stop or token.is_punct or token.text.strip() == "":
            continue
        tokens.append(token.text)
        lemmas.append(token.lemma_)

    # Named Entity Recognition (ERP specific)
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return {
        "original_query": query,
        "normalized_query": normalized,
        "tokens": tokens,
        "lemmas": lemmas,
        "named_entities": entities
    }

# -------------------
# 🔍 Example Test
# -------------------
if __name__ == "__main__":
    test_query = "Where can I see my dispatch note and create a bill for it?"
    output = preprocess_query(test_query)

    # print("🔍 Original Query:", output["original_query"])
    # print("✅ Normalized Query:", output["normalized_query"])
    # print("🧩 Tokens:", output["tokens"])
    # print("🧠 Lemmas:", output["lemmas"])
    # print("🏷️ Named Entities:", output["named_entities"])
