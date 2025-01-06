import re
import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig
import torch

class LoanChatbot:
    def __init__(self):
        print("Initializing LoanChatbot...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.generation_config = GenerationConfig(
            max_new_tokens=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7
        )
        self.user_data = {}

    def generate_greeting(self, user_input):
        templates = [
            f"I understand you're interested in a loan. Let's gather your details.",
            f"I'll help you with your loan application. First, I'll need some information.",
            f"Thanks for your interest in our loan services. Let's get started with your application."
        ]
        return pd.Series(templates).sample().iloc[0]

    def validate_input(self, field_type, value):
        patterns = {
            'email': r'^[\w\.-]+@[\w\.-]+\.\w+$',
            'phone': r'^\d{10}$',
            'dob': r'^\d{2}/\d{2}/\d{4}$',
            'amount': r'^\$?\d+(?:,\d{3})*(?:\.\d{2})?$'
        }
        return bool(re.match(patterns.get(field_type, r'.+'), value))

    def get_next_prompt(self, field):
        prompts = {
            'name': ("Before we proceed, could you tell me your full name?", None),
            'phone': ("What's your contact number?", 'phone'),
            'email': ("Please share your email address:", 'email'),
            'loan_purpose': ("What would you use this loan for?", None),
            'income': ("What's your monthly income?", 'amount'),
            'dob': ("What's your date of birth? (DD/MM/YYYY)", 'dob'),
            'occupation': ("What's your current occupation?", None),
            'address': ("What's your current address?", None),
            'loan_amount': ("What loan amount are you requesting?", 'amount'),
            'promotion_applied': ("Did you use a promotional code? (yes/no)", None),
            'how_heard': ("How did you find out about us?", None),
            'marital_status': ("What is your marital status? (married/single/divorced/widowed)", None),
            'whatsapp_opt_in': ("Would you like updates via WhatsApp? (yes/no)", None),
            'employer_name': ("What is the name of your employer?", None),
            'self_employed': ("Are you self-employed? (yes/no)", None),
            'additional_income': ("Do you have any additional sources of income? (yes/no)", None),
            'commitments': ("Do you have any existing financial commitments?", None),
            'declaration': ("Do you declare all provided information to be true? (yes/no)", None),
            'reference1_name': ("What is the name of your first reference?", None),
            'reference1_relation': ("What is your relation to your first reference?", None),
            'reference1_address': ("What is the address of your first reference?", None),
            'reference1_contact': ("What is the contact number of your first reference?", 'phone'),
            'reference1_occupation': ("What is the occupation of your first reference?", None),
            'reference2_name': ("What is the name of your second reference?", None),
            'reference2_relation': ("What is your relation to your second reference?", None),
            'reference2_address': ("What is the address of your second reference?", None),
            'reference2_contact': ("What is the contact number of your second reference?", 'phone'),
            'reference2_occupation': ("What is the occupation of your second reference?", None)
        }
        return prompts.get(field, (f"Please provide your {field}:", None))

    def handle_document_upload(self):
        file_path = input("Please upload your document (PDF format):\nYou: ")
        if os.path.isfile(file_path) and file_path.endswith('.pdf'):
            return True, "Document received successfully!"
        return False, "Please provide a valid PDF file."

    def extract_features(self, user_input):
        # Feature extraction using regex or simple keywords
        features = {
            'name': re.findall(r'(?<=I am\s)(\w+)', user_input),
            'loan_amount': re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', user_input),
            'loan_purpose': re.findall(r'(car|home|personal|education|medical|business)', user_input, flags=re.IGNORECASE),
            'how_heard': re.findall(r'instagram|facebook|linkedin|google|website', user_input, flags=re.IGNORECASE)
        }
        
        # Example: Clean the results to return only the first match
        return {key: (val[0] if val else None) for key, val in features.items()}

    def process_application(self):
        print("Welcome to the LoanChatbot!")
        print("Please provide some initial information about you.\n")
        initial_input = input("You: ")

        # Extract features from the initial input
        extracted_features = self.extract_features(initial_input)
        
        if extracted_features['name']:
            self.user_data['name'] = extracted_features['name']
            print(f"Extracted name: {self.user_data['name']}")

        if extracted_features['loan_amount']:
            self.user_data['loan_amount'] = extracted_features['loan_amount']
            print(f"Extracted loan amount: {self.user_data['loan_amount']}")

        if extracted_features['loan_purpose']:
            self.user_data['loan_purpose'] = extracted_features['loan_purpose']
            print(f"Extracted loan purpose: {self.user_data['loan_purpose']}")

        if extracted_features['how_heard']:
            self.user_data['how_heard'] = extracted_features['how_heard']
            print(f"Extracted how heard: {self.user_data['how_heard']}")

        print(self.generate_greeting(initial_input))

        # Core application fields
        fields = [
            'name', 'phone', 'email', 'loan_purpose', 'income', 'dob',
            'occupation', 'address', 'loan_amount', 'promotion_applied',
            'how_heard', 'marital_status', 'whatsapp_opt_in', 'employer_name',
            'self_employed', 'additional_income', 'commitments', 'declaration',
            'reference1_name', 'reference1_relation', 'reference1_address',
            'reference1_contact', 'reference1_occupation', 'reference2_name',
            'reference2_relation', 'reference2_address', 'reference2_contact',
            'reference2_occupation'
        ]

        for field in fields:
            while True:
                prompt, validation_type = self.get_next_prompt(field)
                print(f"\n{prompt}")
                response = input("You: ")

                # If user data has been extracted, skip this field and move on
                if field in self.user_data and self.user_data[field]:
                    print(f"Skipping field: {field}, already provided.")
                    break

                if validation_type and not self.validate_input(validation_type, response):
                    print("That doesn't seem right. Please try again.")
                    continue

                self.user_data[field] = response
                break

        # Document collection
        print("\nI'll need some documents to process your application.")
        while not self.handle_document_upload()[0]:
            continue

        # Save application
        df = pd.DataFrame([self.user_data])
        df.to_csv("loan_applications.csv", index=False)

        print("\nThank you! Your loan application has been submitted successfully.")
        print(f"We'll review it and contact you at {self.user_data['email']}.")

if __name__ == "__main__":
    chatbot = LoanChatbot()
    chatbot.process_application()

