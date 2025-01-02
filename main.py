import re
import random
from haystack.nodes import BM25Retriever, FARMReader
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
import spacy
import pandas as pd
import os

# Initialize SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the document store for RAG
document_store = InMemoryDocumentStore()

# Create a synthetic dataset for RAG document store
documents = [
    {"content": "Loan applications require basic personal details, financial details, and reference information."},
    {"content": "Ensure that all fields such as loan purpose, income, and references are filled."},
    {"content": "Membership details and promotional codes can also be included in the loan application."},
    {"content": "Provide accurate and verified contact information, including WhatsApp opt-in preferences."},
    {"content": "References should have a valid relationship, address, and contact details."},
    {"content": "Valid promotional codes are PROMO123, SAVE200, and OFFER500."},
]
document_store.write_documents(documents)

# Initialize Retriever and Reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader("deepset/roberta-base-squad2")

# Create RAG pipeline
qa_pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Predefined valid promotion codes
valid_promo_codes = ["PROMO123", "SAVE200", "OFFER500"]

# Define all required fields
required_fields = [
    'loan_amount', 'promotion_applied', 'loan_purpose', 'how_heard',
    'full_name', 'membership_status', 'account_number',
    'telephone', 'email', 'date_of_birth', 'marital_status',
    'whatsapp_opt_in', 'employer_name', 'self_employed',
    'primary_income', 'additional_income', 'total_income',
    'commitments', 'declaration', 'uploaded_ids', 'uploaded_documents',
    'reference1_name', 'reference1_relation', 'reference1_address',
    'reference1_contact', 'reference1_occupation',
    'reference2_name', 'reference2_relation', 'reference2_address',
    'reference2_contact', 'reference2_occupation'
]

# Dynamic prompts for missing fields
field_prompts = {
    "loan_amount": ["What loan amount are you requesting?"],
    "loan_purpose": ["What are you planning to use the loan for?"],
    "promotion_applied": ["Did you use a promotional code? (yes/no)"],
    "how_heard": ["How did you find out about us?"],
    "full_name": ["What is your full name?"],
    "membership_status": ["Are you a member or non-member?"],
    "account_number": ["What is your account number?"],
    "telephone": ["Please provide your 10-digit contact number."],
    "email": ["What is your email address?"],
    "date_of_birth": ["What is your date of birth? (DD/MM/YYYY)"],
    "marital_status": ["What is your marital status? (married/single/divorced/widowed)"],
    "whatsapp_opt_in": ["Would you like updates via WhatsApp? (yes/no)"],
    "employer_name": ["What is your employer's name?"],
    "self_employed": ["Are you self-employed? (yes/no)"],
    "primary_income": ["What’s your primary monthly income?"],
    "additional_income": ["Do you have any other sources of income?"],
    "total_income": ["What is your total monthly income?"],
    "commitments": ["Do you have any financial commitments?"],
    "declaration": ["Do you confirm that the provided information is true? (yes/no)"],
    "uploaded_ids": ["Please upload your ID document (PDF)."],
    "uploaded_documents": ["Please upload supporting loan documents (PDF)."],
    "reference1_name": ["Who is your first reference?"],
    "reference1_relation": ["What is your relationship with the first reference?"],
    "reference1_address": ["What is the address of your first reference?"],
    "reference1_contact": ["What is the contact number of your first reference?"],
    "reference1_occupation": ["What is the occupation of your first reference?"],
    "reference2_name": ["Who is your second reference?"],
    "reference2_relation": ["What is your relationship with the second reference?"],
    "reference2_address": ["What is the address of your second reference?"],
    "reference2_contact": ["What is the contact number of your second reference?"],
    "reference2_occupation": ["What is the occupation of your second reference?"],
}

# Helper function to split full name
def split_name(full_name):
    name_parts = full_name.split()
    first_name = name_parts[0]
    last_name = name_parts[-1]
    middle_name = " ".join(name_parts[1:-1]) if len(name_parts) > 2 else ""
    return first_name, middle_name, last_name

# Regex-based extraction
def regex_extract(user_input):
    return {
        "email": re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", user_input),
        "telephone": re.search(r"\b\d{10}\b", user_input),
        "loan_amount": re.search(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", user_input),
        "date_of_birth": re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", user_input),
        "marital_status": re.search(r"\b(married|single|divorced|widowed)\b", user_input, re.IGNORECASE),
        "membership_status": re.search(r"\b(member|non-member)\b", user_input, re.IGNORECASE),
        "account_number": re.search(r"\b\d{11,16}\b", user_input)
    }

# Function to validate phone number
def validate_phone_number(phone):
    return bool(re.fullmatch(r"\d{10}", phone))

# Function to validate email
def validate_email(email):
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email))

# Upload document
def upload_document():
    while True:
        file_path = input("Chatbot: Please upload a PDF document:\nYou: ")
        if os.path.isfile(file_path) and file_path.endswith(".pdf"):
            print("Chatbot: Document uploaded successfully!")
            return os.path.basename(file_path)
        print("Chatbot: Invalid file. Try again.")

# Main chatbot function
def chatbot_conversation():
    print("Chatbot: Welcome to the loan application process!")
    user_input = input("Chatbot: Please provide your initial details.\nYou: ")
    user_data = regex_extract(user_input)

    # Full Name extraction and splitting
    if "full_name" not in user_data:
        user_data["full_name"] = input("Chatbot: What is your full name?\nYou: ")
    first_name, middle_name, last_name = split_name(user_data["full_name"])
    user_data["first_name"] = first_name
    user_data["middle_name"] = middle_name
    user_data["last_name"] = last_name

    # Ask for phone number with validation
    while True:
        phone_number = input("Chatbot: Please provide your 10-digit contact number.\nYou: ")
        if validate_phone_number(phone_number):
            user_data["telephone"] = phone_number
            break
        print("Chatbot: Invalid phone number. Please enter a valid 10-digit contact number.")

    # Ask for email with validation
    while True:
        email = input("Chatbot: What is your email address?\nYou: ")
        if validate_email(email):
            user_data["email"] = email
            break
        print("Chatbot: Invalid email address. Please enter a valid email in the format 'example@domain.com'.")

    # Ask for all other required fields dynamically
    for field in required_fields:
        if field not in user_data or not user_data[field]:
            if field == "promotion_applied":
                promo_response = input("Chatbot: Did you use a promotional code? (yes/no)\nYou: ")
                if promo_response.lower() == "yes":
                    promo_code = input("Chatbot: Please enter the promotional code:\nYou: ")
                    user_data["promotion_code"] = promo_code if promo_code in valid_promo_codes else "Invalid"
            elif "uploaded" in field:
                user_data[field] = upload_document()
            else:
                prompt = random.choice(field_prompts[field])
                user_data[field] = input(f"Chatbot: {prompt}\nYou: ")

    # Save to CSV
    df = pd.DataFrame([user_data])
    df.to_csv("loan_applications.csv", index=False)
    print("\nChatbot: Thank you! Your loan application has been submitted.")
    print("Saved details:", df)

# Start chatbot
chatbot_conversation()

