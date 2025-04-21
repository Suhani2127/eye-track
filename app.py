import streamlit as st
import PyPDF2

# --- Title and Instructions ---
st.title("📚 SmartScribe AI - Step 1")
st.subheader("Upload your class notes (PDF or TXT) to extract raw text")

# --- Upload the file ---
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- Display the extracted text ---
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    
    if file_type == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "txt":
        extracted_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")

    st.success("✅ Text extracted successfully!")
    
    with st.expander("🔍 Show Extracted Text"):
        st.write(extracted_text)
