from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader,
    Docx2txtLoader, UnstructuredPowerPointLoader
)
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
import re
import nest_asyncio
from langchain.text_splitter import CharacterTextSplitter
import base64

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import google.generativeai as genai
from PIL import Image
from werkzeug.utils import secure_filename
import os
import json
from fpdf import FPDF
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import shutil
import re
from pydantic import BaseModel
from typing import List
from IPython.display import display, Markdown
import textwrap
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from pydantic import BaseModel
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pathlib
import textwrap
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
import re  # Import regular expression module for hyperlink removal
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
from wordcloud import WordCloud
import collections
import json
import torch
from fpdf import FPDF
from bertopic import BERTopic
import kaleido
import nest_asyncio
import re
import shutil
import os
from fpdf import FPDF
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import List

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

headers = {'Content-Type': 'application/octet-stream'}

if os.getenv("FASTAPI_ENV") == "development":
    nest_asyncio.apply()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your model and other variables
uploaded_file_path = None
document_analyzed = False
summary = None
question_responses = []
api = None
llm = None

safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def format_text(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = text.replace('*', '<br>')
    return text

# Define Pydantic models for requests and responses
class AnalyzeDocumentRequest(BaseModel):
    api_key: str
    prompt: str

    

class AnalyzeDocumentResponse(BaseModel):
    meta: dict
    summary: str
    

    

class AskRequest(BaseModel):
    question: str
    api_key: str

    

class AskResponse(BaseModel):
    meta: dict
    question: str
    result: str
    


# Define Pydantic models for requests and responses
class AnalyzeDocument1Request(BaseModel):
    api_key: str
    custom_question: str

class AnalyzeDocument1Response(BaseModel):
    meta: dict
    plot1_path: str
    response1: str
    plot2_path: str
    response2: str
    pdf_file_path: str
    file_path: str
    columns: str

class MulticlassRequest(BaseModel):
    api_key: str
    custom_question: str
    target_variable: str
    columns_for_analysis: str  # Expecting comma-separated string

class MulticlassResponse(BaseModel):
    meta: dict
    plot3_path: str
    response3: str
    plot4_path: str
    response4: str
    pdf_file_path: str
    file_path: str

class AskRequest1(BaseModel):
    question: str
    api_key: str

class AskResponse1(BaseModel):
    meta: dict
    question: str
    result: str
    



class GetColumn(BaseModel):
    meta: dict
    columns: str
    file_path: str

class AnalyzeDocumentRequest2(BaseModel):
    api_key: str
    target_variable: str
    custom_stopwords: str
    custom_question: str

class AnalyzeDocumentResponse2(BaseModel):
    meta: dict
    sentiment_plot_path: str
    topic_plot_path: str
    topic_plot_path1: str
    topic_plot_path2: str
    wordcloud_positive: str
    gemini_response_pos: str
    wordcloud_neutral: str
    gemini_response_neu: str
    wordcloud_negative: str
    gemini_response_neg: str
    bigram_positive: str
    gemini_response_pos1: str
    bigram_neutral: str
    gemini_response_neu1: str
    bigram_negative: str
    gemini_response_neg1: str
    unigram_positive: str
    gemini_response_pos2: str
    unigram_neutral: str
    gemini_response_neu2: str
    unigram_negative: str
    gemini_response_neg2: str
    pdf_file_path: str
    file_path: str


class AskRequest2(BaseModel):
    question: str
    api_key: str

class AskResponse2(BaseModel):
    meta: dict
    question: str
    result: str
    

# Route for analyzing documents
@app.post("/py/v1", response_model=AnalyzeDocumentResponse)
async def analyze_document(
    api_key: str = Form(...),
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    global uploaded_file_path, document_analyzed, summary, api, llm
    loader = None

    try:
        # Initialize or update API key and models
        api = api_key
        genai.configure(api_key=api)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api)

        # Save the uploaded file
        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(await file.read())  # Using async file read

        # Determine the file type and load accordingly
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(uploaded_file_path)
        elif file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements", encoding="utf8")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(uploaded_file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(uploaded_file_path)
        elif file_extension == ".mp3":
            # Process audio files differently
            audio_file = genai.upload_file(path=uploaded_file_path)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            prompt = f"{prompt}"
            response = model.generate_content([prompt, audio_file], safety_settings=safety_settings)
            summary = format_text(response.text)
            document_analyzed = True
            return AnalyzeDocumentResponse(meta={"status": "success", "code": 200}, summary=summary)

        # If no loader is set, raise an exception
        if loader is None:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

        docs = loader.load()
        prompt_template = PromptTemplate.from_template(
            f"""
            You are an expert at analyzing and interpreting information. Below is the provided content:

            {{text}}

            Instructions:
            1. Carefully analyze the provided content and identify key numerical data and their corresponding attributes.
            2. If the question asks for specific comparisons like "highest," "lowest," or other numerical operations:
            - First, identify all relevant numerical values in the data (e.g., sales prices, quantities, dates).
            - Then, compare these values and determine the highest, lowest, or any other metric requested by the question.
            - For example, if the question asks for the highest value, identify the highest number and return the corresponding data (e.g., "Order ID with the highest sale price").
            - If the question asks for a total or sum, calculate that from the relevant data.
            3. If the question asks for a summary:
            - Provide a concise and clear summary of the most important information from the data, ensuring it includes key figures or findings from the data.

            Question: {prompt}

            Provide only the final answer in your response, based on the data analysis.
            """
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        response = stuff_chain.invoke(docs)
        summary = format_text(response["output_text"])
        document_analyzed = True

        return AnalyzeDocumentResponse(meta={"status": "success", "code": 200}, summary=summary)

    except Exception as e:
        print(f"An error occurred during document analysis: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="An error occurred during document analysis.")

# Route for answering questions
@app.post("/py/v1/ask", response_model=AskResponse)
async def ask_question(
    request: Request,
    api_key: str = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...),
):
    global uploaded_file_path, document_analyzed, summary, api, llm
    loader = None

    try:
        # Initialize or update API key and models
        api = api_key
        genai.configure(api_key=api)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api)

        # Save the uploaded file
        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(await file.read())  # Using async file read

        # Determine the file type and load accordingly
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(uploaded_file_path)
        elif file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".docx":
            loader = Docx2txtLoader(uploaded_file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(uploaded_file_path)
        elif file_extension == ".mp3":
            audio_file = genai.upload_file(path=uploaded_file_path)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            latest_conversation = request.cookies.get("latest_question_response", "")
            prompt = "Answer the question based on the speech: " + question + (f" Latest conversation: {latest_conversation}" if latest_conversation else "")
            
            # Generate response based on audio input
            response = model.generate_content([prompt, audio_file], safety_settings=safety_settings)
            current_response = response.text
            current_question = f"You asked: {question}"

            # Save the latest question and response to the session
            question_responses.append((current_question, current_response))

            # Use the summary generated from the MP3 content as text
            text = current_response

            # Set the Google API key
            os.environ["GOOGLE_API_KEY"] = api

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            # Generate embeddings for the chunks
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            document_search = FAISS.from_texts([chunk.page_content for chunk in chunks], embeddings)

            if document_search:
                query_embedding = embeddings.embed_query(question)
                results = document_search.similarity_search_by_vector(query_embedding, k=1)

                if results:
                    current_response = format_text(results[0].page_content)
                else:
                    current_response = "No matching document found in the database."
            else:
                current_response = "Vector database not initialized."

            return AskResponse(meta={"status": "success", "code": 200}, question=question, result=current_response)
        

        # If no loader is set, raise an exception
        if loader is None:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)  # Pass the list of Document objects

        # Generate embeddings for the chunks
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        document_search = FAISS.from_texts([chunk.page_content for chunk in chunks], embeddings)

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f"{question} Answer the question based on the following:\n\"{retrieved_texts}\"\n:" +
                (f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}" if latest_conversation else "")
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            response_chain = llm_chain1.invoke({"text": retrieved_texts})
            summary1 = response_chain["text"]

            # Return the response
            return AskResponse(meta={"status": "success", "code": 200}, question=question, result=summary1)
        else:
            return AskResponse(meta={"status": "success", "code": 200}, question=question, result="No relevant results found.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    






### TABULAR ANALYSIS ----------------------------------------------------------------







safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]



sns.set_theme(color_codes=True)
uploaded_df = None
question_responses = []






def format_text(text):
    # Replace **text** with <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace any remaining * with <br>
    text = text.replace('*', '<br>')
    return text

def clean_data(df):
    # Step 1: Clean currency-related columns
    for col in df.columns:
        if any(x in col.lower() for x in ['value', 'price', 'cost', 'amount']):
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '').str.replace('£', '').str.replace('€', '').replace('[^\d.-]', '', regex=True).astype(float)
    
    # Step 2: Drop columns with more than 25% missing values
    null_percentage = df.isnull().sum() / len(df)
    columns_to_drop = null_percentage[null_percentage > 0.25].index
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Step 3: Fill missing values for remaining columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if null_percentage[col] <= 0.25:
                if df[col].dtype in ['float64', 'int64']:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
    
    # Step 4: Convert object-type columns to lowercase
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    # Step 5: Drop columns with only one unique value
    unique_value_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=unique_value_columns, inplace=True)

    return df




def clean_data2(df):
    for col in df.columns:
        if 'value' in col or 'price' in col or 'cost' in col or 'amount' in col or 'Value' in col or 'Price' in col or 'Cost' in col or 'Amount' in col:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '')
                df[col] = df[col].str.replace('£', '')
                df[col] = df[col].str.replace('€', '')
                df[col] = df[col].replace('[^\d.-]', '', regex=True).astype(float)


    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    return df



def generate_plot(df, plot_path, plot_type):
    df = clean_data(df)
    excluded_words = ["name", "postal", "date", "phone", "address", "code", "id"]

    if plot_type == 'countplot':
        cat_vars = [col for col in df.select_dtypes(include='object').columns
                    if all(word not in col.lower() for word in excluded_words) and df[col].nunique() > 1]

        for col in cat_vars:
            if df[col].nunique() > 10:
                top_categories = df[col].value_counts().index[:10]
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')

        num_cols = len(cat_vars)
        num_rows = (num_cols + 1) // 2
        fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        for i, var in enumerate(cat_vars):
            category_counts = df[var].value_counts()
            top_values = category_counts.index[:10][::-1]
            filtered_df = df.copy()
            filtered_df[var] = pd.Categorical(filtered_df[var], categories=top_values, ordered=True)
            sns.countplot(x=var, data=filtered_df, order=top_values, ax=axs[i])
            axs[i].set_title(var)
            axs[i].tick_params(axis='x', rotation=30)

            total = len(filtered_df[var])
            for p in axs[i].patches:
                height = p.get_height()
                axs[i].annotate(f'{height / total:.1%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

            sample_size = filtered_df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

    elif plot_type == 'histplot':
        num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns
                    if all(word not in col.lower() for word in excluded_words)]
        num_cols = len(num_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(nrows=num_rows, ncols=min(3, num_cols), figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        plot_index = 0

        for i, var in enumerate(num_vars):
            if len(df[var].unique()) == len(df):
                fig.delaxes(axs[plot_index])
            else:
                sns.histplot(df[var], ax=axs[plot_index], kde=True, stat="percent")
                axs[plot_index].set_title(var)
                axs[plot_index].set_xlabel('')

            sample_size = df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

            plot_index += 1

        for i in range(plot_index, len(axs)):
            fig.delaxes(axs[i])

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


@app.post("/py/v1/result", response_model=AnalyzeDocument1Response)
async def result(api_key: str = Form(...), 
                 file: UploadFile = File(...), 
                 custom_question: str = Form(...)):
    global uploaded_df

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read the file depending on the extension
    if uploaded_filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif uploaded_filename.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    columns = df.columns.tolist()

    # Function to generate Gemini response based on the plot
    def generate_gemini_response(plot_path):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        img = Image.open(plot_path)
        response = model.generate_content([custom_question + " Analyze the data insights from the chart.", img])
        response.resolve()
        return response.text

    try:
        # Generate plots
        plot1_path = generate_plot(df, 'static/plot1.png', 'countplot')
        plot2_path = generate_plot(df, 'static/plot2.png', 'histplot')

        # Generate Gemini responses
        response1 = generate_gemini_response(plot1_path)
        response2 = generate_gemini_response(plot2_path)

        uploaded_df = df

        # Generate PDF
        def safe_encode(text):
            try:
                return text.encode('latin1', errors='replace').decode('latin1')  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"
            
        pdf = FPDF()
        pdf.set_font("Arial", size=12)

        pdf.add_page()
        pdf.cell(200, 10, txt="Single Countplot Barchart", ln=True, align='C')
        pdf.image(plot1_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Single Countplot Barchart Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response1))

        pdf.add_page()
        pdf.cell(200, 10, txt="Single Histoplot", ln=True, align='C')
        pdf.image(plot2_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Single Histoplot Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response2))

        pdf_file_path = os.path.join("static", "output.pdf")
        pdf.output(pdf_file_path)

        # Upload files to external endpoint
        url = "https://app.goarif.co/api/v1/Attachment/Upload/Paython"

        # Upload CSV
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(url, files={"file": (file_path, f, "text/csv")})

        # Upload Plot1
        with open(plot1_path, "rb") as f:
            plot1_url = requests.post(url, files={"file": ("plot1.png", f, "image/png")})

        # Upload Plot2
        with open(plot2_path, "rb") as f:
            plot2_url = requests.post(url, files={"file": ("plot2.png", f, "image/png")})

        # Upload PDF
        with open(pdf_file_path, "rb") as f:
            pdf_url = requests.post(url, files={"file": ("output.pdf", f, "application/pdf")})

        # Return the response with data and the upload result
        return AnalyzeDocument1Response(
            meta={
                "status": "success", 
                "code": 200, 
            },
            file_path=uploaded_file_url.text,
            plot1_path=plot1_url.text,
            plot2_path=plot2_url.text,
            pdf_file_path=pdf_url.text,
            response1=response1,
            response2=response2,
            columns=", ".join(columns)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








@app.post("/py/v1/multiclass", response_model=MulticlassResponse)
async def multiclass(
    request: Request,
    target_variable: str = Form(...),
    custom_question: str = Form(...),
    api_key: str = Form(...),
    file: UploadFile = File(...),
    columns_for_analysis: str = Form(...),  # Changed to str to handle CSV string input
):
    global document_analyzed

    try:
        # Read the file content into a DataFrame
        if file.filename.endswith('.csv'):
            # Load CSV file into DataFrame
            df = pd.read_csv(file.file, encoding='utf-8')
            

        elif file.filename.endswith('.xlsx'):
            # Load Excel file into DataFrame
            df = pd.read_excel(file.file)
            

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Process the columns_for_analysis
        columns_for_analysis_list = [col.strip() for col in columns_for_analysis.split(',')]

        # Ensure the columns exist in the DataFrame
        missing_cols = [col for col in columns_for_analysis_list if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found in the dataset: {', '.join(missing_cols)}")

        # Select the target variable and columns for analysis from the DataFrame
        if target_variable not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target variable '{target_variable}' not found in the dataset")

        target_variable_data = df[target_variable]
        columns_for_analysis_data = df[columns_for_analysis_list]

        # Concatenate target variable and columns for analysis into a single DataFrame
        df = pd.concat([target_variable_data, columns_for_analysis_data], axis=1)

        # Clean the data (if needed)
        df = clean_data2(df)

        # Generate visualizations

        # Multiclass Barplot
        excluded_words = ["name", "postal", "date", "phone", "address", "id"]

        # Get the names of all columns with data type 'object' (categorical variables)
        cat_vars = [col for col in df.select_dtypes(include=['object']).columns
                    if all(word not in col.lower() for word in excluded_words)]

        # Exclude the target variable from the list if it exists in cat_vars
        if target_variable in cat_vars:
            cat_vars.remove(target_variable)

        # Create a figure with subplots, but only include the required number of subplots
        num_cols = len(cat_vars)
        num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a count plot for each categorical variable
        for i, var in enumerate(cat_vars):
            top_categories = df[var].value_counts().nlargest(5).index
            filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]  # Exclude rows with NaN values in the variable

            # Replace less frequent categories with "Other" if there are more than 5 unique values
            if df[var].nunique() > 5:
                other_categories = df[var].value_counts().index[5:]
                filtered_df[var] = filtered_df[var].apply(lambda x: x if x in top_categories else 'Other')

            sns.countplot(x=var, hue=target_variable, data=filtered_df, ax=axs[i], stat="percent")
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)

            # Change y-axis label to represent percentage
            axs[i].set_ylabel('Percentage')

            # Annotate the subplot with sample size
            sample_size = df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        # Remove any remaining blank subplots
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

        plt.xticks(rotation=45)
        plt.tight_layout()
        plot3_path = "static/multiclass_barplot.png"
        plt.savefig(plot3_path)
        plt.close(fig)

        # Multiclass Histplot
        int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
        int_vars = [col for col in int_vars if col != target_variable]

        # Create a figure with subplots
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a histogram for each integer variable with hue=target_variable
        for i, var in enumerate(int_vars):
            sns.histplot(data=df, x=var, hue=target_variable, kde=True, ax=axs[i], stat="percent")
            axs[i].set_title(var)

            # Annotate the subplot with sample size
            sample_size = df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        # Remove any extra empty subplots if needed
        if num_cols < len(axs):
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])

        fig.tight_layout()
        plt.xticks(rotation=45)
        plot4_path = "static/multiclass_histplot.png"
        plt.savefig(plot4_path)
        plt.close(fig)

        # Google Gemini Responses
        genai.configure(api_key=api_key)

        # Response for the barplot
        img_barplot = Image.open(plot3_path)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response3 = format_text(model.generate_content([custom_question, img_barplot]).text)

        # Response for the histplot
        img_histplot = Image.open(plot4_path)
        response4 = format_text(model.generate_content([custom_question, img_histplot]).text)

        document_analyzed = True

        # Create a dictionary to store the outputs
        outputs = {
            "multiBarchart_visualization": plot3_path,
            "gemini_response3": response3,
            "multiHistoplot_visualization": plot4_path,
            "gemini_response4": response4
        }

        # Save the dictionary as a JSON file
        with open("output1.json", "w") as outfile:
            json.dump(outputs, outfile)

        def safe_encode(text):
            try:
                return text.encode('latin1', errors='replace').decode('latin1')  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"

        # Generate PDF with the results
        pdf = FPDF()
        pdf.set_font("Arial", size=12)

        # Add content to the PDF
        # Multiclass Countplot Barchart and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Countplot Barchart", ln=True, align='C')
        pdf.image(plot3_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Countplot Barchart Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response3))

        # Multiclass Histplot and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Histplot", ln=True, align='C')
        pdf.image(plot4_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Histplot Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response4))

        pdf_output_path = 'static/analysis_report_complete.pdf'
        pdf.output(pdf_output_path)
        pdf_file_path = pdf_output_path.replace("\\", "/")



        uploaded_filename = secure_filename(file.filename)
        file_path = os.path.join("static", uploaded_filename)

        # Save the uploaded file
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Upload files to external endpoint
        url = "https://app.goarif.co/api/v1/Attachment/Upload/Paython"

        # Upload CSV
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(url, files={"file": (file_path, f, "text/csv")})

        # Upload Plot1
        with open(plot3_path, "rb") as f:
            plot3_url = requests.post(url, files={"file": ("multiclass_barplot.png", f, "image/png")})

        # Upload Plot2
        with open(plot4_path, "rb") as f:
            plot4_url = requests.post(url, files={"file": ("multiclass_histplot.png", f, "image/png")})

        # Upload PDF
        with open(pdf_file_path, "rb") as f:
            pdf_url = requests.post(url, files={"file": ("analysis_report_complete.pdf", f, "application/pdf")})


        return MulticlassResponse(
            meta={"status": "success", "code": 200},
            plot3_path=plot3_url.text,
            plot4_path=plot4_url.text,
            response3=response3,
            response4=response4,
            pdf_file_path=pdf_url.text,
            file_path=uploaded_file_url.text
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Route for answering questions
@app.post("/py/v1/ask1", response_model=AskResponse1)
async def ask_question(
    request: Request,
    api_key: str = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...)
):
    global uploaded_file_path, document_analyzed, summary, api, llm
    
    loader = None

    try:

        # Initialize the LLM model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(file.file.read())
        # Determine the file extension and select the appropriate loader
        loader = None
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Load and process the document
        try:
            docs = loader.load()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

        # Combine document text
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api_key

        # Initialize embeddings and create FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)  # Pass the list of Document objects
        document_search = FAISS.from_texts([chunk.page_content for chunk in chunks], embeddings)

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f"{question} Answer the question based on the following:\n\"{text}\"\n:" +
                (f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}" if latest_conversation else "")
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            try:
                response_chain = llm_chain1.invoke({"text": text})
                summary1 = response_chain["text"]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error invoking LLMChain: {str(e)}")

            # Generate embeddings for the summary
            try:
                summary_embedding = embeddings.embed_query(summary1)
                document_search = FAISS.from_texts([summary1], embeddings)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

            # Perform a search on the FAISS vector database
            try:
                if document_search:
                    query_embedding = embeddings.embed_query(question)
                    results = document_search.similarity_search_by_vector(query_embedding, k=1)

                    if results:
                        current_response = format_text(results[0].page_content)
                    else:
                        current_response = "No matching document found in the database."
                else:
                    current_response = "Vector database not initialized."
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during similarity search: {str(e)}")
        else:
            current_response = "No relevant results found."

        # Append the question and response from FAISS search
        current_question = f"You asked: {question}"
        question_responses.append((current_question, current_response))

        # Save all results to output_summary.json
        save_to_json(question_responses)

        return AskResponse1(meta={"status": "success", "code": 200}, question=question, result=current_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



def save_to_json(question_responses):
    outputs = {
        "question_responses": question_responses
    }
    with open("output_summary.json", "w") as outfile:
        json.dump(outputs, outfile)









## SENTIMENT ANALYSIS ----------------------------------------------------------------



question_responses = []
document_analyzed = False







# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Create stopword remover
stop_factory = StopWordRemoverFactory()
more_stopword = ['yg', 'yang', 'aku', 'gw', 'gua', 'gue']
data = stop_factory.get_stop_words() + more_stopword

# Define hyperlink pattern for removal
hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
number_pattern = re.compile(r'\b\d+\b')

emoticon_pattern = re.compile(u'('
    u'\ud83c[\udf00-\udfff]|'
    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
    u'[\u2600-\u26FF\u2700-\u27BF])+', 
    re.UNICODE)


@app.post("/py/v1/process", response_model=GetColumn)
async def process_file(request: Request, file: UploadFile = File(...)):
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load DataFrame based on file type
    file_extension = os.path.splitext(file.filename)[1]
    try:
        if file_extension == '.csv':
            # Load CSV file into DataFrame
            df = pd.read_csv(file_path, delimiter=",")
        elif file_extension in ['.xls', '.xlsx']:
            # Load Excel file into DataFrame
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        # Get columns of the DataFrame
        columns = df.columns.tolist()

        # Upload CSV
        url = "https://app.goarif.co/api/v1/Attachment/Upload/Paython"
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(url, files={"file": (file_path, f, "text/csv")})

        return GetColumn(
            meta={"status": "success", "code": 200},
            columns=", ".join(columns),
            file_path=uploaded_file_url.text  # Return the external file URL
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/py/v1/analyze", response_model=AnalyzeDocumentResponse2)
async def analyze(
    request: Request,
    api_key: str = Form(...),
    target_variable: str = Form(...),
    custom_stopwords: str = Form(""),
    file: UploadFile = File(...),
    hf_token: str = Form(...),
    custom_question: str = Form("")
):
    global df
    # Read the uploaded CSV file

    os.environ["HF_TOKEN"] = hf_token

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load DataFrame based on file type
    file_extension = os.path.splitext(file.filename)[1]
    try:
        if file_extension == '.csv':
            # Load CSV file into DataFrame
            df = pd.read_csv(file_path, delimiter=",")
            
            

        elif file_extension in ['.xls', '.xlsx']:
            # Load Excel file into DataFrame
            df = pd.read_excel(file_path)
            
            

        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")


        if target_variable not in df.columns:
            return "Selected target variable does not exist in the dataset."

        add_stopwords = ['the', 'of', 'is', 'a', 'in', 'https', 'yg', 'gua', 'gue', 'lo', 'lu', 'gw']
        custom_stopword_list = [word.strip() for word in custom_stopwords.split(',')]
        all_stopwords = data + custom_stopword_list + add_stopwords

        # Remove hyperlinks, emoticons, numbers, and stopwords
        hyperlink_pattern = r'https?://\S+|www\.\S+'
        emoticon_pattern = r'[:;=X][oO\-]?[D\)\]\(\]/\\OpP]'
        number_pattern = r'\b\d+\b'

        df[target_variable] = df[target_variable].astype(str)
        df['cleaned_text'] = df[target_variable].str.replace(hyperlink_pattern, '', regex=True)
        df['cleaned_text'] = df['cleaned_text'].str.replace(emoticon_pattern, '', regex=True)
        df['cleaned_text'] = df['cleaned_text'].str.replace(number_pattern, '', regex=True)
        for stopword in all_stopwords:
            df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() != stopword]))

        # Perform stopwords removal
        df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(
            [stemmer.stem(word) for word in stop_factory.create_stop_word_remover().remove(x).split()
            if word.lower() not in all_stopwords]
        ))

        # Perform Sentiment Analysis
        pretrained = "indonesia-bert-sentiment-classification"
        model = AutoModelForSequenceClassification.from_pretrained(pretrained)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

        def analyze_sentiment(text):
            result = sentiment_analysis(text)
            label = label_index[result[0]['label']]
            score = result[0]['score']
            return pd.Series({'sentiment_label': label, 'sentiment_score': score})

        df[['sentiment_label', 'sentiment_score']] = df['cleaned_text'].apply(analyze_sentiment)

        # Count the occurrences of each sentiment label
        sentiment_counts = df['sentiment_label'].value_counts()

        # Plot a bar chart using seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Count')
        sentiment_plot_path = 'static/sentiment_distribution.png'
        plt.savefig(sentiment_plot_path)

        # Convert plots to base64 strings
        with open(sentiment_plot_path, "rb") as image_file:
            sentiment_plot_bytes = base64.b64encode(image_file.read()).decode('utf-8')

        model = BERTopic(verbose=True)
        model.fit(df['cleaned_text'])
        topics, probabilities = model.transform(df['cleaned_text'])
        fig = model.visualize_barchart()
        fig.write_image('static/barchart.png')
        topic_plot_path = 'static/barchart.png'

        fig1 = model.visualize_hierarchy()
        fig1.write_image('static/hierarchy.png')
        topic_plot_path1 = 'static/hierarchy.png'

        topic_distr, _ = model.approximate_distribution(df['cleaned_text'], min_similarity=0)
        fig2 = model.visualize_distribution(topic_distr[0])
        fig2.write_image('static/dist.png')
        topic_plot_path2 = 'static/dist.png'


        
        

        # Generate sentiment analysis results table
        analysis_results = df.to_html(classes='data')

        # Concatenate Cleaned text
        positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['cleaned_text'])
        negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['cleaned_text'])
        neutral_text = ' '.join(df[df['sentiment_label'] == 'neutral']['cleaned_text'])

        # Create WordCloud Positive
        wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='Set2', background_color='white'
        ).generate(positive_text)

        wordcloud_positive = "static/wordcloud_positive.png"
        wordcloud.to_file(wordcloud_positive)

        # Use Google Gemini API to generate content based on the uploaded image
        img = PIL.Image.open(wordcloud_positive)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        try:
            response = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img])
            response.resolve()
            gemini_response_pos = format_text(response.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_pos = "Error: Failed to generate content with Gemini API."

        # Create WordCloud Neutral
        wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='Set2', background_color='white'
        ).generate(neutral_text)

        wordcloud_neutral = "static/wordcloud_neutral.png"
        wordcloud.to_file(wordcloud_neutral)

        img = PIL.Image.open(wordcloud_neutral)
        try:
            response = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img])
            response.resolve()
            gemini_response_neu = format_text(response.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neu = "Error: Failed to generate content with Gemini API."

        # Create WordCloud Negative
        wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='Set2', background_color='white'
        ).generate(negative_text)

        wordcloud_negative = "static/wordcloud_negative.png"
        wordcloud.to_file(wordcloud_negative)

        img = PIL.Image.open(wordcloud_negative)
        try:
            response = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img])
            response.resolve()
            gemini_response_neg = format_text(response.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neg = "Error: Failed to generate content with Gemini API."
        


        # Convert plots to base64 strings
        with open(wordcloud_positive, "rb") as image_file:
            wordcloud_positive_bytes = base64.b64encode(image_file.read()).decode('utf-8')
        # Convert plots to base64 strings
        with open(wordcloud_neutral, "rb") as image_file:
            wordcloud_neutral_bytes = base64.b64encode(image_file.read()).decode('utf-8')
        # Convert plots to base64 strings
        with open(wordcloud_negative, "rb") as image_file:
            wordcloud_negative_bytes = base64.b64encode(image_file.read()).decode('utf-8')



        # Bigram Positive
        words1 = positive_text.split()
        bigrams = list(zip(words1, words1[1:]))
        bigram_counts = collections.Counter(bigrams)
        top_bigrams = dict(bigram_counts.most_common(10))

        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)
        plt.xlabel('Count')
        plt.ylabel('Bigram Words')
        plt.title("Top 10 Bigram Positive Sentiment")

        bigram_positive = "static/bigram_positive.png"
        plt.savefig(bigram_positive)
        

        img1 = PIL.Image.open(bigram_positive)
        try:
            response1 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img1])
            response1.resolve()
            gemini_response_pos1 = format_text(response1.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_pos1 = "Error: Failed to generate content with Gemini API."

        # Bigram Neutral
        words2 = neutral_text.split()
        bigrams = list(zip(words2, words2[1:]))
        bigram_counts = collections.Counter(bigrams)
        top_bigrams = dict(bigram_counts.most_common(10))

        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)
        plt.xlabel('Count')
        plt.ylabel('Bigram Words')
        plt.title("Top 10 Bigram Neutral Sentiment")

        bigram_neutral = "static/bigram_neutral.png"
        plt.savefig(bigram_neutral)
        

        img2 = PIL.Image.open(bigram_neutral)
        try:
            response2 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img2])
            response2.resolve()
            gemini_response_neu1 = format_text(response2.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neu1 = "Error: Failed to generate content with Gemini API."

        # Bigram Negative
        words3 = negative_text.split()
        bigrams = list(zip(words3, words3[1:]))
        bigram_counts = collections.Counter(bigrams)
        top_bigrams = dict(bigram_counts.most_common(10))

        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)
        plt.xlabel('Count')
        plt.ylabel('Bigram Words')
        plt.title("Top 10 Bigram Negative Sentiment")

        bigram_negative = "static/bigram_negative.png"
        plt.savefig(bigram_negative)
        

        img3 = PIL.Image.open(bigram_negative)
        try:
            response3 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img3])
            response3.resolve()
            gemini_response_neg1 = format_text(response3.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neg1 = "Error: Failed to generate content with Gemini API."


        
        # Convert plots to base64 strings
        with open(bigram_positive, "rb") as image_file:
            bigram_positive_bytes = base64.b64encode(image_file.read()).decode('utf-8')
        # Convert plots to base64 strings
        with open(bigram_neutral, "rb") as image_file:
            bigram_neutral_bytes = base64.b64encode(image_file.read()).decode('utf-8')
        # Convert plots to base64 strings
        with open(bigram_negative, "rb") as image_file:
           bigram_negative_bytes = base64.b64encode(image_file.read()).decode('utf-8')


            
        # Unigram Positive
        words2 = positive_text.split()

        # Count the occurrences of each word
        word_counts = collections.Counter(words2)

        # Get top 10 words
        top_words = dict(word_counts.most_common(10))

        # Create bar chart
        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_words)), list(top_words.values()), align='center')  # Horizontal bar chart
        plt.yticks(range(len(top_words)), list(top_words.keys()), rotation=0)  # Swapping y-axis and x-axis
        plt.xlabel('Count')  # Changed the label to Count
        plt.ylabel('Words')  # Changed the label to Words
        plt.title("Top 10 Unigram Positive Sentiment")
        # Save the unigram image
        unigram_positive = "static/unigram_positive.png"
        # Save the entire plot as a PNG
        plt.savefig(unigram_positive)
        # Show the plot
        

        # Use Google Gemini API to generate content based on the bigram image
        img1 = PIL.Image.open(unigram_positive)
        try:
            response1 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img1])
            response1.resolve()
            gemini_response_pos2 = format_text(response1.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_pos2 = "Error: Failed to generate content with Gemini API."
                    


        # Unigram Neutral
        words2 = neutral_text.split()

        # Count the occurrences of each word
        word_counts = collections.Counter(words2)

        # Get top 10 words
        top_words = dict(word_counts.most_common(10))

        # Create bar chart
        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_words)), list(top_words.values()), align='center')  # Horizontal bar chart
        plt.yticks(range(len(top_words)), list(top_words.keys()), rotation=0)  # Swapping y-axis and x-axis
        plt.xlabel('Count')  # Changed the label to Count
        plt.ylabel('Words')  # Changed the label to Words
        plt.title("Top 10 Unigram Neutral Sentiment")
        # Save the unigram image
        unigram_neutral = "static/unigram_neutral.png"
        # Save the entire plot as a PNG
        plt.savefig(unigram_neutral)
        # Show the plot
        

        # Use Google Gemini API to generate content based on the bigram image
        img1 = PIL.Image.open(unigram_neutral)
        try:
            response1 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img1])
            response1.resolve()
            gemini_response_neu2 = format_text(response1.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neu2 = "Error: Failed to generate content with Gemini API."




        # Unigram Negative
        words2 = negative_text.split()

        # Count the occurrences of each word
        word_counts = collections.Counter(words2)

        # Get top 10 words
        top_words = dict(word_counts.most_common(10))

        # Create bar chart
        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_words)), list(top_words.values()), align='center')  # Horizontal bar chart
        plt.yticks(range(len(top_words)), list(top_words.keys()), rotation=0)  # Swapping y-axis and x-axis
        plt.xlabel('Count')  # Changed the label to Count
        plt.ylabel('Words')  # Changed the label to Words
        plt.title("Top 10 Unigram Negative Sentiment")
        # Save the unigram image
        unigram_negative = "static/unigram_negative.png"
        # Save the entire plot as a PNG
        plt.savefig(unigram_negative)
        # Show the plot
        

        # Use Google Gemini API to generate content based on the bigram image
        img1 = PIL.Image.open(unigram_negative)
        try:
            response1 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img1])
            response1.resolve()
            gemini_response_neg2 = format_text(response1.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neg2 = "Error: Failed to generate content with Gemini API."
            
        document_analyzed = True

        # Function to handle encoding to latin1
        def safe_encode(text):
            try:
                return text.encode('latin1', errors='replace').decode('latin1')  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"


        # Convert plots to base64 strings
        with open(unigram_positive, "rb") as image_file:
            unigram_positive_bytes = base64.b64encode(image_file.read()).decode('utf-8')
        # Convert plots to base64 strings
        with open(unigram_neutral, "rb") as image_file:
            unigram_neutral_bytes = base64.b64encode(image_file.read()).decode('utf-8')
        # Convert plots to base64 strings
        with open(unigram_negative, "rb") as image_file:
           unigram_negative_bytes = base64.b64encode(image_file.read()).decode('utf-8')




        # Generate PDF with the results
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align='C')

        # Sentiment Distribution Plot
        pdf.image(sentiment_plot_path, x=10, y=30, w=190)
        pdf.ln(100)

        pdf.add_page()
        pdf.cell(200, 10, txt="Topic Modelling Barchart", ln=True, align='C')
        pdf.image(topic_plot_path, x=10, y=30, w=190)

        pdf.add_page()
        pdf.cell(200, 10, txt="Topic Modelling Hierarchy", ln=True, align='C')
        pdf.image(topic_plot_path1, x=10, y=30, w=190)

        pdf.add_page()
        pdf.cell(200, 10, txt="Topic Modelling Distribution", ln=True, align='C')
        pdf.image(topic_plot_path2, x=10, y=30, w=190)
                    

        # Positive WordCloud and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive WordCloud", ln=True, align='C')
        pdf.image(wordcloud_positive, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive WordCloud Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_pos))

        # Neutral WordCloud and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral WordCloud", ln=True, align='C')
        pdf.image(wordcloud_neutral, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral WordCloud Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neu))

        # Negative WordCloud and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative WordCloud", ln=True, align='C')
        pdf.image(wordcloud_negative, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative WordCloud Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neg))

        # Positive Bigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive Bigram Sentiment", ln=True, align='C')
        pdf.image(bigram_positive, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive Bigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_pos1))

        # Neutral Bigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral Bigram Sentiment", ln=True, align='C')
        pdf.image(bigram_neutral, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral Bigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neu1))

        # Negative Bigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative Bigram Sentiment", ln=True, align='C')
        pdf.image(bigram_negative, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative Bigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neg1))

        # Positive Unigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive Unigram Sentiment", ln=True, align='C')
        pdf.image(unigram_positive, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive Unigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_pos2))

        # Neutral Unigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral Unigram Sentiment", ln=True, align='C')
        pdf.image(unigram_neutral, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral Unigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neu2))

        # Negative Unigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative Unigram Sentiment", ln=True, align='C')
        pdf.image(unigram_negative, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative Unigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neg2))
        
        pdf_file_path = os.path.join("static", "sentiment.pdf")
        pdf.output(pdf_file_path)

        pdf_file_path = os.path.join("static", "sentiment.pdf")
        pdf_file_path = pdf_file_path.replace("\\", "/")

       
            
        # Upload files to external endpoint
        url = "https://app.goarif.co/api/v1/Attachment/Upload/Paython"

        # Upload CSV
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(url, files={"file": (file_path, f, "text/csv")})
        
        # Upload Plot1
        with open(sentiment_plot_path, "rb") as f:
            sentiment_url = requests.post(url, files={"file": ("sentiment_distribution.png", f, "image/png")})

        # Upload Plot1
        with open(topic_plot_path, "rb") as f:
            topic_plot_url1 = requests.post(url, files={"file": ("barchart.png", f, "image/png")})

        # Upload Plot2
        with open(topic_plot_path1, "rb") as f:
            topic_plot_url2 = requests.post(url, files={"file": ("hierarchy.png", f, "image/png")})
        
        # Upload Plot2
        with open(topic_plot_path2, "rb") as f:
            topic_plot_url3 = requests.post(url, files={"file": ("dist.png", f, "image/png")})
        
        # Upload wordcloud
        with open(wordcloud_positive, "rb") as f:
            wordcloud_positive_url = requests.post(url, files={"file": ("wordcloud_positive.png", f, "image/png")})

        # Upload Plot2
        with open(wordcloud_neutral, "rb") as f:
            wordcloud_neutral_url = requests.post(url, files={"file": ("wordcloud_neutral.png", f, "image/png")})
        
        # Upload Plot2
        with open(wordcloud_negative, "rb") as f:
            wordcloud_negative_url = requests.post(url, files={"file": ("wordcloud_negative.png", f, "image/png")})

        # Upload bigram
        with open(bigram_positive, "rb") as f:
            bigram_positive_url = requests.post(url, files={"file": ("bigram_positive.png", f, "image/png")})

        # Upload Plot2
        with open(bigram_neutral, "rb") as f:
            bigram_neutral_url = requests.post(url, files={"file": ("bigram_neutral.png", f, "image/png")})
        
        # Upload Plot2
        with open(bigram_negative, "rb") as f:
            bigram_negative_url = requests.post(url, files={"file": ("bigram_negative.png", f, "image/png")})
        
         # Upload unigram
        with open(unigram_positive, "rb") as f:
            unigram_positive_url = requests.post(url, files={"file": ("unigram_positive.png", f, "image/png")})

        # Upload Plot2
        with open(unigram_neutral, "rb") as f:
            unigram_neutral_url = requests.post(url, files={"file": ("unigram_neutral.png", f, "image/png")})
        
        # Upload Plot2
        with open(unigram_negative, "rb") as f:
            unigram_negative_url = requests.post(url, files={"file": ("unigram_negative.png", f, "image/png")})

        # Upload PDF
        with open(pdf_file_path, "rb") as f:
            pdf_url = requests.post(url, files={"file": ("sentiment.pdf", f, "application/pdf")})
        

        return AnalyzeDocumentResponse2(
            meta={"status": "success", "code": 200},
            sentiment_plot_path=sentiment_url.text,
            topic_plot_path=topic_plot_url1.text,
            topic_plot_path1=topic_plot_url2.text,
            topic_plot_path2=topic_plot_url3.text,
            wordcloud_positive=wordcloud_positive_url.text,
            gemini_response_pos=gemini_response_pos,
            wordcloud_neutral=wordcloud_neutral_url.text,
            gemini_response_neu=gemini_response_neu,
            wordcloud_negative=wordcloud_negative_url.text,
            gemini_response_neg=gemini_response_neg,
            bigram_positive=bigram_positive_url.text,
            gemini_response_pos1=gemini_response_pos1,
            bigram_neutral=bigram_neutral_url.text,
            gemini_response_neu1=gemini_response_neu1,
            bigram_negative=bigram_negative_url.text,
            gemini_response_neg1=gemini_response_neg1,
            unigram_positive=unigram_positive_url.text,
            gemini_response_pos2=gemini_response_pos2,
            unigram_neutral=unigram_neutral_url.text,
            gemini_response_neu2=gemini_response_neu2,
            unigram_negative=unigram_negative_url.text,
            gemini_response_neg2=gemini_response_neg2,
            pdf_file_path=pdf_url.text,
            file_path=uploaded_file_url.text
        )


    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))




# Route for answering questions
@app.post("/py/v1/ask2", response_model=AskResponse2)
async def ask_question(
    request: Request,
    api_key: str = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...)
):
    global uploaded_file_path, document_analyzed, summary, api, llm
    
    loader = None

    try:

        # Initialize the LLM model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(file.file.read())
        # Determine the file extension and select the appropriate loader
        loader = None
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Load and process the document
        try:
            docs = loader.load()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

        # Combine document text
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api_key

        # Initialize embeddings and create FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)  # Pass the list of Document objects
        document_search = FAISS.from_texts([chunk.page_content for chunk in chunks], embeddings)

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f"{question} Answer the question based on the following:\n\"{text}\"\n:" +
                (f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}" if latest_conversation else "")
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            try:
                response_chain = llm_chain1.invoke({"text": text})
                summary1 = response_chain["text"]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error invoking LLMChain: {str(e)}")

            # Generate embeddings for the summary
            try:
                summary_embedding = embeddings.embed_query(summary1)
                document_search = FAISS.from_texts([summary1], embeddings)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

            # Perform a search on the FAISS vector database
            try:
                if document_search:
                    query_embedding = embeddings.embed_query(question)
                    results = document_search.similarity_search_by_vector(query_embedding, k=1)

                    if results:
                        current_response = results[0].page_content
                    else:
                        current_response = "No matching document found in the database."
                else:
                    current_response = "Vector database not initialized."
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during similarity search: {str(e)}")
        else:
            current_response = "No relevant results found."

        # Append the question and response from FAISS search
        current_question = f"You asked: {question}"
        question_responses.append((current_question, current_response))

        # Save all results to output_summary.json
        save_to_json(question_responses)

        return AskResponse2(meta={"status": "success", "code": 200}, question=question, result=current_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



def save_to_json(question_responses):
    outputs = {
        "question_responses": question_responses
    }
    with open("output_summary.json", "w") as outfile:
        json.dump(outputs, outfile)
    




@app.get("/download_pdf")
async def download_pdf():
    pdf_output_path = 'static/analysis_report.pdf'
    return FileResponse(pdf_output_path, filename="analysis_report.pdf", media_type='application/pdf')



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
