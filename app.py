import streamlit as st
from main import LoanChatbot
import pandas as pd

# Initialize the chatbot
chatbot = LoanChatbot()

# Streamlit Layout and styling
st.set_page_config(page_title="Loan Chatbot", layout="centered")
st.title("Loan Application Chatbot")

# Show greeting message
st.write(chatbot.generate_greeting("start"))

# Initialize the session state to store user responses and conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Function to handle user input and chatbot responses
def get_chatbot_response(user_input):
    # Extract features from the user input
    extracted_features = chatbot.extract_features(user_input)

    # Update user data in chatbot
    for key, value in extracted_features.items():
        if value:
            chatbot.user_data[key] = value

    # Show the next question based on the field to be filled
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

    # If there's any field left to fill, ask the next question
    for field in fields:
        if field not in chatbot.user_data:
            prompt, validation_type = chatbot.get_next_prompt(field)
            st.session_state.conversation.append(f"Chatbot: {prompt}")
            return prompt  # Return the question to ask next

    return "All required details are collected!"

# Get and display the conversation
if st.session_state.conversation:
    for message in st.session_state.conversation:
        st.write(message)

# User Input Section
user_input = st.text_input("You:", "")
if user_input:
    st.session_state.conversation.append(f"You: {user_input}")
    next_question = get_chatbot_response(user_input)

    # Show the chatbot response
    st.session_state.conversation.append(f"Chatbot: {next_question}")

# Handling document uploads after all fields are filled
if len(chatbot.user_data) == len(fields):  # Ensure all fields are collected
    if chatbot.handle_document_upload()[0]:
        st.session_state.conversation.append("Chatbot: Document received successfully.")

# Display the CSV link after the application is saved
if len(chatbot.user_data) == len(fields):
    df = pd.DataFrame([chatbot.user_data])
    df.to_csv("loan_applications.csv", index=False)
    st.session_state.conversation.append("Chatbot: Your details are saved successfully.")
    st.download_button(
        label="Download Loan Application CSV",
        data=df.to_csv(index=False),
        file_name="loan_application.csv",
        mime="text/csv"
    )
