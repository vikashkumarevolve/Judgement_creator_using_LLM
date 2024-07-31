import streamlit as st
from PyPDF2 import PdfReader
import openai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import textwrap  # Added for text wrapping
from concurrent.futures import ThreadPoolExecutor
import faiss  # Vector database library

# Initialize the OpenAI client with the API key
api_key = ""  # Replace with your actual API key
from openai import OpenAI
client = OpenAI(api_key=api_key)

MAX_TOKENS = 16000  # Adjust this value based on the model's maximum context length and desired completion length
JUDGMENT_FOLDER_PATH = "E:\\Gen AI Project\\LawGPT Documents"  # Predefined path to the existing judgment folder

# Initialize FAISS index
d = 1536  # Dimension of embeddings from text-embedding-ada-002
index = faiss.IndexFlatL2(d)

# Dictionary to map filenames to embeddings and texts
embedding_text_map = {}

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() or ""
    return text

def extract_key_excerpts(judgment_texts):
    key_excerpts = []
    def extract_excerpt(text, idx):
        token_count = len(text.split()) + 500  # Approximate token count including the completion
        if token_count > MAX_TOKENS:
            return None, idx

        prompt = f"Extract the key excerpts from the following legal judgment:\n\n{text}\n\nKey Excerpts:"
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a legal assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            if response and response.choices:
                return response.choices[0].message.content.strip(), idx
        except Exception as e:
            return None, idx
        return None, idx

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_excerpt, text, idx) for idx, text in enumerate(judgment_texts)]
        for future in futures:
            excerpt, idx = future.result()
            if excerpt:
                key_excerpts.append(excerpt)
            else:
                st.warning(f"Document {idx} exceeds the maximum token limit and will be skipped.")
    return key_excerpts

def generate_embeddings(texts):
    embeddings = []
    def generate_embedding(text, idx):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=[text]  # Note the input is a list
            )
            if response and response.data:
                return response.data[0].embedding, idx
        except Exception as e:
            return None, idx
        return None, idx

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_embedding, text, idx) for idx, text in enumerate(texts)]
        for future in futures:
            embedding, idx = future.result()
            if embedding:
                embeddings.append(embedding)
            else:
                st.error(f"Failed to generate embeddings for text {idx}.")
                return None
    return np.array(embeddings)

def match_relevant_judgments_faiss(new_petition_embedding, top_n=5):
    distances, indices = index.search(np.array([new_petition_embedding]), top_n)
    matched_excerpts = [embedding_text_map[i]['excerpt'] for i in indices[0]]
    matched_filenames = [embedding_text_map[i]['filename'] for i in indices[0]]
    return matched_excerpts, matched_filenames

def generate_judgment(response_texts, new_petition_text, matched_excerpts, relevant_cases):
    prompt = f"""
    Given the following previous legal responses and key excerpts from judgments:

    New Petition:
    {new_petition_text}

    Responses: {' '.join(response_texts)}

    Key Excerpts from Previous Judgments: {' '.join(matched_excerpts)}

    Relevant Cases from the Supreme Court: {' '.join(relevant_cases)}

    Generate a legal judgment for the new petition:
    {new_petition_text}

    The judgment should be consistent with the tone, terminology, and style of the previous responses and judgments. 
    Cite the relevant Supreme Court cases in the legal analysis to ensure that the analysis is correct and based on established law.
    """

    st.text_area("Prompt sent to GPT-4o", prompt, height=300)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.7,
        )

        st.json(response)  # Display the raw response for debugging

        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            st.error("Failed to generate a judgment. The API did not return a valid response.")
            return None
    except Exception as e:
        st.error(f"An error occurred while generating the judgment: {e}")
        return None

def extract_relevant_cases(judgment_texts, section_number):
    cases = []
    for text in judgment_texts:
        # Check for lines that mention the Supreme Court and the relevant section number
        case_lines = [
            line for line in text.split('\n') 
            if "Supreme Court" in line and section_number in line
        ]
        cases.extend(case_lines)
    return cases

def read_existing_judgments(folder_path):
    judgment_texts = []
    judgment_filenames = []
    def read_judgment(file_path):
        with open(file_path, 'rb') as file:
            return extract_text_from_pdf(file), file_path

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_judgment, os.path.join(root, file_name)) 
                   for root, _, files in os.walk(folder_path) 
                   for file_name in files if file_name.endswith('.pdf')]
        for future in futures:
            text, file_path = future.result()
            judgment_texts.append(text)
            judgment_filenames.append(file_path)
    return judgment_texts, judgment_filenames

def save_text_to_pdf(text, filename):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)
    width, height = letter
    lines = text.split('\n')
    y = height - 40
    for line in lines:
        if line.strip() == "":  # Add extra space for empty lines indicating new paragraphs
            y -= 14  # Add extra space between paragraphs
        else:
            for wrapped_line in textwrap.wrap(line, width=90):  # Adjust width to control text wrapping
                pdf.drawString(40, y, wrapped_line)
                y -= 14
                if y < 40:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 12)
                    y = height - 40
            y -= 14  # Add extra space after each line
    pdf.save()

    buffer.seek(0)
    return buffer

st.title("Legal Judgment Generator")
st.write("Automate the generation of legal judgments for new petitions.")

# Dropdown for selecting IPC section
ipc_section = st.selectbox("Select IPC section:", ["Section 482", "Section 307", "Section 302", "Section 420", "Section 138"])

uploaded_response_files = st.file_uploader("Upload Response PDFs", accept_multiple_files=True)
uploaded_petition_file = st.file_uploader("Upload New Petition PDF", accept_multiple_files=False)

if st.button("Generate Judgment"):
    if uploaded_response_files and uploaded_petition_file:
        with ThreadPoolExecutor() as executor:
            response_futures = [executor.submit(extract_text_from_pdf, file) for file in uploaded_response_files]
            response_texts = [future.result() for future in response_futures]
        new_petition_text = extract_text_from_pdf(uploaded_petition_file)

        # Adjust the folder path based on the selected IPC section
        section_folder_mapping = {
            "Section 482": "S.482 CrPC",
            "Section 307": "S. 307 IPC",
            "Section 302": "S. 302 IPC",
            "Section 420": "S. 420 IPC",
            "Section 138": "S. 138 NI Act"
        }
        section_folder = section_folder_mapping.get(ipc_section, f"S. {ipc_section.split()[-1]} IPC")
        section_path = os.path.join(JUDGMENT_FOLDER_PATH, section_folder)

        existing_judgments, judgment_filenames = read_existing_judgments(section_path)

        if not existing_judgments:
            st.error(f"No judgments found for {ipc_section} in the path {section_path}")
        else:
            key_excerpts = extract_key_excerpts(existing_judgments)
            if key_excerpts:
                new_petition_embedding = generate_embeddings([new_petition_text])
                if new_petition_embedding is not None:
                    new_petition_embedding = new_petition_embedding[0]

                    judgment_embeddings = generate_embeddings(key_excerpts)
                    if judgment_embeddings is not None:
                        # Add embeddings and excerpts to FAISS index
                        for idx, embedding in enumerate(judgment_embeddings):
                            index.add(np.array([embedding]))
                            embedding_text_map[len(embedding_text_map)] = {
                                'excerpt': key_excerpts[idx],
                                'filename': judgment_filenames[idx]
                            }

                        matched_excerpts, matched_filenames = match_relevant_judgments_faiss(new_petition_embedding)

                        section_number = ipc_section.split()[1]  # Extract section number from selected IPC section
                        relevant_cases = extract_relevant_cases(existing_judgments, section_number)
                        generated_judgment = generate_judgment(response_texts, new_petition_text, matched_excerpts, relevant_cases)

                        if generated_judgment:
                            st.success("Generated Judgment:")
                            st.text_area("Generated Judgment", generated_judgment, height=400)

                            pdf_buffer = save_text_to_pdf(generated_judgment, "Generated_Judgment.pdf")
                            st.download_button(
                                label="Download Judgment as PDF",
                                data=pdf_buffer,
                                file_name="Generated_Judgment.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("Failed to generate the judgment.")
            else:
                st.error("Failed to extract key excerpts from the existing judgments.")
    else:
        st.error("Please upload both response PDFs and a new petition PDF.")
