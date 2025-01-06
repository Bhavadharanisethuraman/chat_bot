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

# Initialize the session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "user_data" not in st.session_state:
    st.session_state.user_data = {}  # Store user responses here
if "current_field" not in st.session_state:
    st.session_state.current_field = None  # Track the current question being asked
if "fields" not in st.session_state:
    st.session_state.fields = [
        'name', 'phone', 'email', 'loan_purpose', 'income', 'dob',
        'occupation', 'address', 'loan_amount', 'promotion_applied',
        'how_heard', 'marital_status', 'whatsapp_opt_in', 'employer_name',
        'self_employed', 'additional_income', 'commitments', 'declaration',
        'reference1_name', 'reference1_relation', 'reference1_address',
        'reference1_contact', 'reference1_occupation', 'reference2_name',
        'reference2_relation', 'reference2_address', 'reference2_contact',
        'reference2_occupation'
    ]
if "next_question" not in st.session_state:
    st.session_state.next_question = None

# Function to get the chatbot's next response
def get_chatbot_response(user_input):
    current_field = st.session_state.current_field

    # Save the response for the current field without validation for phone
    if current_field:
        if current_field == "phone":
            st.session_state.user_data[current_field] = user_input  # Accept any input for phone
        else:
            extracted_features = chatbot.extract_features(user_input)
            if current_field in extracted_features:
                st.session_state.user_data[current_field] = extracted_features[current_field]
            else:
                # Invalid input, repeat the same question
                return f"Invalid input for {current_field}. {chatbot.get_next_prompt(current_field)[0]}"

    # Check for the next field to fill
    for field in st.session_state.fields:
        if field not in st.session_state.user_data:
            st.session_state.current_field = field
            prompt, validation_type = chatbot.get_next_prompt(field)
            return prompt

    # If all fields are collected
    st.session_state.current_field = None
    return "All required details are collected! Your application is complete."

# Display the conversation
if st.session_state.conversation:
    for message in st.session_state.conversation:
        st.write(message)

# User Input Section
placeholder = st.empty()  # Create a placeholder for the input widget

# Use a unique key for each text input widget
unique_key = f"user_input_{len(st.session_state.conversation)}"
with placeholder:
    user_input = st.text_input("You:", key=unique_key)

if user_input:
    # Append user's response to the conversation
    st.session_state.conversation.append(f"You: {user_input}")

    # Get chatbot's next response
    next_question = get_chatbot_response(user_input)

    # Append chatbot's response to the conversation
    st.session_state.conversation.append(f"Chatbot: {next_question}")

    # Clear the input field by recreating it
    placeholder.empty()
    placeholder.text_input("You:", key=f"user_input_{len(st.session_state.conversation)}")

    # If the application is complete, save the data to a CSV file
    if next_question == "All required details are collected! Your application is complete.":
        df = pd.DataFrame([st.session_state.user_data])
        df.to_csv("loan_applications.csv", index=False)
        st.session_state.conversation.append("Chatbot: Your details are saved successfully.")
        st.session_state.conversation.append("Chatbot: Download your application details below:")
        st.download_button(
            label="Download Loan Application CSV",
            data=df.to_csv(index=False),
            file_name="loan_application.csv",
            mime="text/csv"
        )

# Automatically scroll to the latest message in the conversation
if st.session_state.conversation:
    st.write("---")
    st.write("### Chat Conversation:")
    st.write("\n".join(st.session_state.conversation))
