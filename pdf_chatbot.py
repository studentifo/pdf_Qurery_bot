import streamlit as st
import pdfplumber
import requests
import time

# Set Hugging Face API Token and Endpoint
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
API_TOKEN = "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"  # Replace with your token
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)  # Handle empty pages gracefully
    return text.strip()

# Function to query Hugging Face API with error handling
def query_huggingface_api(prompt, retries=3):
    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

        if response.status_code == 200:
            try:
                # Ensure we properly handle list-based responses
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response generated.")
                return "Unexpected response format."
            except Exception as e:
                st.error(f"Error parsing response: {e}")
                return ""

        elif response.status_code == 503:
            st.warning(f"Model is loading. Retrying in 10 seconds... (Attempt {attempt + 1})")
            time.sleep(10)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            break  # Exit on other errors

    return "Sorry, the model could not generate a response at this time."

# Main function for the Streamlit app
def main():
    st.title("AI PDF Chatbot")

    # PDF file uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.success("PDF uploaded successfully!")

        # Extract and display the PDF content
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted PDF Content", pdf_text[:5000], height=200)  # Limit displayed content to 5000 chars

        # User input text area
        user_input = st.text_area("You:", "")

        if user_input:
            # Prepare query with user input and PDF content
            with st.spinner("Generating response..."):
                prompt = f"Based on the document, answer this: {user_input}"
                bot_response = query_huggingface_api(prompt)

            # Display the bot's response
            st.write(f"**Bot:** {bot_response}")

if __name__ == "__main__":
    main()
