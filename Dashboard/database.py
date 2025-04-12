import sqlalchemy as db
import random

# Create SQLite Database
engine = db.create_engine('sqlite:///IDMS_ERP.db')
connection = engine.connect()
metadata = db.MetaData()

# Define Tables
modules_table = db.Table('Modules', metadata,
    db.Column('module_id', db.Integer, primary_key=True),
    db.Column('module_name', db.String, nullable=False),
    db.Column('description', db.String)
)

faqs_table = db.Table('FAQs', metadata,
    db.Column('faq_id', db.Integer, primary_key=True),
    db.Column('question', db.String, nullable=False),
    db.Column('answer', db.String, nullable=False),
    db.Column('module_id', db.Integer, db.ForeignKey('Modules.module_id')),
    db.Column('category', db.String, nullable=False),
    db.Column('times_asked', db.Integer, nullable=False)
)

# Create Tables
metadata.create_all(engine)

# Insert Data for Modules
modules = [
    {'module_name': 'Accounts & Finance', 'description': 'Tracks financial transactions, payments, GST compliance, and reporting.'},
    {'module_name': 'Sales & NPD', 'description': 'Manages quotations, sales orders, invoices, and dispatches.'},
    {'module_name': 'Purchase', 'description': 'Handles procurement, supplier management, and purchase orders.'},
    {'module_name': 'Stores', 'description': 'Maintains stock levels, material movements, and inventory tracking.'},
    {'module_name': 'Production', 'description': 'Controls manufacturing processes, job work, and raw material consumption.'},
    {'module_name': 'Quality', 'description': 'Ensures compliance through inspections and material validation.'},
    {'module_name': 'Dispatch & Logistics', 'description': 'Organizes shipments, transport partners, and delivery tracking.'},
    {'module_name': 'HR & Admin', 'description': 'Manages workforce, payroll, and employee records.'}
]

connection.execute(modules_table.insert(), modules)

# Define GST-Related Questions
gst_questions = [
    'What is GST and why is it important for businesses?',
    'How does IDMS help in GST compliance?',
    'What are the different types of GST in IDMS?',
    'What is the role of HSN & SAC codes in IDMS?',
    'How does E-Invoicing work in IDMS?',
    'When is an E-Way Bill required?',
    'What is the Reverse Charge Mechanism (RCM) in GST?',
    'How does IDMS handle tax invoice vs. proforma invoice?',
    'Can IDMS generate GST returns automatically?',
    'How does IDMS help in reconciling GST mismatches?',
    'What is GSTR-1 and how does IDMS help in filing it?',
    'What is GSTR-3B and how does IDMS assist in its filing?',
    'How does IDMS handle ITC (Input Tax Credit)?',
    'What happens if there is a GST mismatch in IDMS?',
    'How does IDMS manage GST for inter-state vs. intra-state transactions?',
    'How are HSN-wise summary reports generated in IDMS?',
    'What happens if a sales invoice is cancelled after GST has been filed?',
    'How does IDMS handle GST-exempt and zero-rated supplies?',
    'How does IDMS automate GST payments?',
    'How does IDMS generate audit reports for GST compliance?'
]

# Populate FAQs
faqs = []
for module_id in range(1, 9):
    for question in gst_questions:
        times_asked = random.randint(1, 20)
        for _ in range(times_asked):
            faqs.append({
                'question': question,
                'answer': f'Sample answer for {question.lower()}.',
                'module_id': module_id,
                'category': 'Taxation',
                'times_asked': times_asked
            })

# Limit to 200-300 total entries
faqs = faqs[:random.randint(200, 300)]
connection.execute(faqs_table.insert(), faqs)

print("IDMS ERP database with multiple modules and GST-related FAQs populated successfully.")
