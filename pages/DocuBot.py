import os
import streamlit as st
import xml.etree.ElementTree as ET
import PyPDF2
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, download_loader, LLMPredictor, ServiceContext
import openai
from pathlib import Path
from typing import List
import json
from langchain import OpenAI
# from dotenv import load_dotenv
import io

# load_dotenv()
# os.getenv("API_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]
PDFReader = download_loader("PDFReader")
loader = PDFReader()

# Initialize JSONReader
JSONReader = download_loader("JSONReader")
loader_json = JSONReader()

# Define the data directory path
DATA_DIR = "data"

def display_pdf(pdf_file):
    with open(pdf_file, "rb") as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            with st.expander(f"Page {page_num+1}"):
                st.write(page.extractText())

def display_json(json_file):
    with open(json_file, "r") as f:
        json_data = json.load(f)
        st.json(json_data)

def delete_directory(directory_path):
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(directory_path)
    st.success(f"Directory {directory_path} deleted successfully!")

def save_uploaded_file(uploaded_file, file_dir):
    with open(os.path.join(file_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

def xml_to_json(xml_str):
    data = ET.parse(io.StringIO(xml_str))
    root = data.getroot()
    def _parse(node):
        json_node = dict()
        if len(list(node)) == 0:
            return node.text
        else:
            for child in node:
                if child.tag not in json_node:
                    json_node[child.tag] = _parse(child)
                else:
                    if type(json_node[child.tag]) is list:
                        json_node[child.tag].append(_parse(child))
                    else:
                        json_node[child.tag] = [json_node[child.tag], _parse(child)]
        return json_node

    return _parse(root)

def process_data(documents):
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-davinci-003", max_tokens=1000))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    vector_index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    return vector_index


def main():
    st.title("DocuBOT Admin")
    container = st.container()
    with container:
        tab1, tab2 = st.tabs(["Upload PDF", "Upload XML"])
    
    with tab1:
        uploaded_file_pdf = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file_pdf is not None:
            file_name = uploaded_file_pdf.name
        
            dir_path = os.path.join(DATA_DIR, file_name)
            os.makedirs(dir_path, exist_ok=True)
            save_uploaded_file(uploaded_file_pdf, dir_path)
            # tab1.success("It would take a while to index the books, please wait..!")
            
            pdf_filename = uploaded_file_pdf.name
            documents = loader.load_data(
                file=Path(f"{os.path.join(dir_path, pdf_filename)}"))

            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-curie-001", max_tokens=2800))
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
            index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

            # index.save_to_disk(os.path.join(dir_path, os.path.splitext(pdf_filename)[0] + ".json"))
            index.storage_context.persist(persist_dir=dir_path)
                
            tab1.success("Index created successfully!")

        DATA_DIR = "/data"
        directories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.endswith(".pdf")]

        colms = st.columns((4, 1, 1))
        fields = ["Name", 'View', 'Delete']
        for col, field_name in zip(colms, fields):
            col.subheader(field_name)

        i = 1
        for dir_name in directories:
            i += 1
            col1, col2, col3 = st.columns((4, 1, 1))
            col1.caption(dir_name)
            file_path = os.path.join(DATA_DIR, dir_name, dir_name)
            if os.path.isfile(file_path):
                col2.button("View", key=file_path, on_click=display_pdf, args=(file_path,))
                delete_status = True
            else:
                col2.write("N/A")
                delete_status = False
            button_type = "Delete" if delete_status else "Gone"
            button_phold_pdf = col3.empty()
            do_action = button_phold_pdf.button(
                button_type, key=i, on_click=delete_directory, args=(os.path.join(DATA_DIR, dir_name),))

           
    with tab2:
        uploaded_file = st.file_uploader("Upload a XML file", type="xml")
        if uploaded_file is not None:
            file_name = uploaded_file.name
            directory_path = Path('data') / file_name
            directory_path.mkdir(parents=True, exist_ok=True)

            bytes_data = uploaded_file.read()
            str_data = bytes_data.decode("utf-8")
            json_data = xml_to_json(str_data)

            json_file_path = directory_path / 'data.json'
            with json_file_path.open('w') as f:
                json.dump(json_data, f)
            # tab2.success("It would take a while to index the books, please wait..!")
            
            documents = loader_json.load_data(json_file_path)
            vector_index = process_data(documents)
            vector_index.storage_context.persist(persist_dir=directory_path)
            
            tab2.success("Vector Index created successfully")

        # directories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        directories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.endswith(".xml")]

        colms = st.columns((4, 1, 1))
        fields = ["Name", 'View', 'Delete']
        for col, field_name in zip(colms, fields):
            col.subheader(field_name)

        j = 1
        for dir_name in directories:
            j += 1
            col1, col2, col3 = st.columns((4, 1, 1))
            col1.caption(dir_name)
            file_path = os.path.join(DATA_DIR, dir_name, "data.json")
            if os.path.isfile(file_path):
                col2.button("View", key=file_path, on_click=display_json, args=(file_path,))
                delete_status = True
            else:
                col2.write("N/A")
                delete_status = False
            button_type = "Delete" if delete_status else "Gone"
            button_phold = col3.empty()
            do_action = button_phold.button(
                button_type, key=f"{dir_name}j", on_click=delete_directory, args=(os.path.join(DATA_DIR, dir_name),))

if __name__ == "__main__":

    DATA_DIR = "data"
    main()


# import os
# import streamlit as st
# import PyPDF2
# from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, download_loader, LLMPredictor, ServiceContext
# import openai
# from pathlib import Path
# from typing import List
# import json
# from langchain import OpenAI
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("API_KEY")
# PDFReader = download_loader("PDFReader")
# loader = PDFReader()

# # Define the data directory path
# DATA_DIR = "data"
# DB_FILE = "db.json"
# # Create the data directory if it doesn't exist
# if not os.path.exists(DATA_DIR):
#     os.mkdir(DATA_DIR)


# def load_users_dicts() -> List[dict]:
#     if Path(DB_FILE).is_file():
#         with open(DB_FILE, "r") as f:
#             users_data = json.load(f)
#         return users_data
#     else:
#         return []

# # Define a function to display the contents of a PDF file


# def display_pdf(DATA_DIR, pdf_file):
#     with open(os.path.join(DATA_DIR, pdf_file), "rb") as f:
#         pdf_reader = PyPDF2.PdfFileReader(f)
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             with st.expander(f"Page {page_num+1}"):
#                 st.write(page.extractText())

# # Define a function to delete a PDF file and its corresponding JSON index file


# def delete_file(DATA_DIR, file_name):
#     pdf_path = os.path.join(DATA_DIR, file_name)
#     json_path = os.path.join(
#         DATA_DIR, os.path.splitext(file_name)[0] + ".json")
#     if os.path.exists(pdf_path):
#         os.remove(pdf_path)
#         st.success(f"File {file_name} deleted successfully!")
#     else:
#         st.error(f"File {file_name} not found!")
#     if os.path.exists(json_path):
#         os.remove(json_path)

# # Define a function to save the uploaded file to the data directory


# def save_uploaded_file(uploaded_file):
#     with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
#         f.write(uploaded_file.getbuffer())

# # Define the Streamlit app


# def main():
#     st.title("DocuBOT Admin")

#     # Create a file uploader widget
#     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

#     # Check if a file was uploaded
#     if uploaded_file is not None:
#         # Save the uploaded file to the data directory
#         save_uploaded_file(uploaded_file)
#         st.success("It would take a while to index the books, please wait..!")

#     # Create a button to create the index
#     # if st.button("Create Index"):
#         # Get the filename of the uploaded PDF
#         pdf_filename = uploaded_file.name

#         # Load the documents from the data directory
#         documents = loader.load_data(
#             file=Path(f"{os.path.join(DATA_DIR, uploaded_file.name)}"))

#         # Create the index from the documents
#         llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-curie-001", max_tokens=2800))
#         service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
#         index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        
#         # index = GPTVectorStoreIndex.from_documents(documents)

#         # Save the index to the data directory with the same name as the PDF
#         index.save_to_disk(os.path.join(
#             DATA_DIR, os.path.splitext(pdf_filename)[0] + ".json"))
#         st.success("Index created successfully!")

#     # Get a list of files in the directory
#     files = os.listdir(DATA_DIR)

#     # Filter out the JSON index files
#     files = [f for f in files if not f.endswith(".json")]

#     colms = st.columns((4, 1, 1))

#     fields = ["Name", 'View', 'Delete']
#     for col, field_name in zip(colms, fields):
#         # header
#         col.subheader(field_name)

#     i = 1
#     for Name in files:
#         i += 1
#         col1, col2, col3 = st.columns((4, 1, 1))
#         # col1.write(x)  # index
#         col1.caption(Name)  # email
#         if Name.endswith(".pdf"):
#             col2.button("View", key=Name, on_click=display_pdf,
#                         args=(DATA_DIR, Name))  # unique ID
#             delete_status = True
#         else:
#             col2.write("N/A")
#             delete_status = False
#         button_type = "Delete" if delete_status else "Gone"
#         button_phold = col3.empty()  # create a placeholder
#         do_action = button_phold.button(
#             button_type, key=i, on_click=delete_file, args=(DATA_DIR, Name))

#     # users_dicts = load_users_dicts()
#     # st.table(users_dicts)


# if __name__ == "__main__":
#     main()

# import os
# import streamlit as st
# import xml.etree.ElementTree as ET
# import PyPDF2
# from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, download_loader, LLMPredictor, ServiceContext
# import openai
# from pathlib import Path
# from typing import List
# import json
# from langchain import OpenAI
# from dotenv import load_dotenv
# import io

# load_dotenv()
# # os.getenv("API_KEY")
# openai.api_key = os.environ["OPENAI_API_KEY"]
# PDFReader = download_loader("PDFReader")
# loader = PDFReader()

# # Define the data directory path
# DATA_DIR = "data"

# def display_pdf(pdf_file):
#     with open(pdf_file, "rb") as f:
#         pdf_reader = PyPDF2.PdfFileReader(f)
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             with st.expander(f"Page {page_num+1}"):
#                 st.write(page.extractText())

# def delete_directory(directory_path):
#     for root, dirs, files in os.walk(directory_path, topdown=False):
#         for name in files:
#             os.remove(os.path.join(root, name))
#         for name in dirs:
#             os.rmdir(os.path.join(root, name))
#     os.rmdir(directory_path)
#     st.success(f"Directory {directory_path} deleted successfully!")

# def save_uploaded_file(uploaded_file, file_dir):
#     with open(os.path.join(file_dir, uploaded_file.name), "wb") as f:
#         f.write(uploaded_file.getbuffer())

# def main():
#     st.title("DocuBOT Admin")
#     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

#     if uploaded_file is not None:
#         dir_path = os.path.join(DATA_DIR, uploaded_file.name)
#         os.makedirs(dir_path, exist_ok=True)
#         save_uploaded_file(uploaded_file, dir_path)
#         st.success("It would take a while to index the books, please wait..!")
        
#         pdf_filename = uploaded_file.name
#         documents = loader.load_data(
#             file=Path(f"{os.path.join(dir_path, pdf_filename)}"))

#         llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-curie-001", max_tokens=2800))
#         service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
#         index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

#         # index.save_to_disk(os.path.join(dir_path, os.path.splitext(pdf_filename)[0] + ".json"))
#         index.storage_context.persist(persist_dir=dir_path)
            
#         st.success("Index created successfully!")

#     directories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

#     colms = st.columns((4, 1, 1))
#     fields = ["Name", 'View', 'Delete']
#     for col, field_name in zip(colms, fields):
#         col.subheader(field_name)

#     i = 1
#     for dir_name in directories:
#         i += 1
#         col1, col2, col3 = st.columns((4, 1, 1))
#         col1.caption(dir_name)
#         file_path = os.path.join(DATA_DIR, dir_name, f"{dir_name}.pdf")
#         if os.path.isfile(file_path):
#             col2.button("View", key=file_path, on_click=display_pdf, args=(file_path,))
#             delete_status = True
#         else:
#             col2.write("N/A")
#             delete_status = False
#         button_type = "Delete" if delete_status else "Gone"
#         button_phold = col3.empty()
#         do_action = button_phold.button(
#             button_type, key=i, on_click=delete_directory, args=(os.path.join(DATA_DIR, dir_name),))



