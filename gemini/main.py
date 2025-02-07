from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
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

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from PIL import Image
from werkzeug.utils import secure_filename
import json
from fpdf import FPDF
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import shutil

from fastapi import Request

sns.set_theme(color_codes=True)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import PIL.Image
from wordcloud import WordCloud
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
import requests
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

headers = {"Content-Type": "application/octet-stream"}

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


def format_text(text):
    # Replace **text** with <b>text</b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<br><b>\1</b>", text)
    # Replace any remaining * with <br>
    text = text.replace("*", "<br>")
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
    analysis_results: str
    wordcloud_positive: str
    gemini_response_pos: str
    wordcloud_negative: str
    gemini_response_neg: str
    response_result: str
    pdf_file_path: str
    file_path: str


class AskRequest2(BaseModel):
    question: str
    api_key: str


class AskResponse2(BaseModel):
    meta: dict
    question: str
    result: str


class GetResult(BaseModel):
    meta: dict
    columns: str
    file_path: str


class AnalyzeDocumentRequest3(BaseModel):
    api_key: str
    custom_question: str
    clustering_columns: str


class AnalyzeDocumentResponse3(BaseModel):
    meta: dict
    table_data: str
    recommendation_table_html: str
    summary: str
    file_path: str


class GetResult1(BaseModel):
    meta: dict
    columns: str
    file_path: str


# Define Pydantic models for requests and responses
class AnalyzeDocument1Request(BaseModel):
    api_key: str
    question: str
    target_variable: str


class AnalyzeDocumentResponse4(BaseModel):
    meta: dict
    file_path: str
    table_data: str  # List[Dict[str, Any]]
    summary: str


# Route for analyzing documents
@app.post("/py/v1", response_model=AnalyzeDocumentResponse)
async def analyze_document(
    api_key: str = Form(...), prompt: str = Form(...), file: UploadFile = File(...)
):
    global uploaded_file_path, document_analyzed, summary, api, llm
    loader = None

    try:
        # Initialize or update API key and models
        api = api_key
        genai.configure(api_key=api)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=api)

        # Save the uploaded file
        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(await file.read())  # Using async file read

        # Determine the file type and load accordingly
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(uploaded_file_path)
        elif file_extension == ".csv":
            loader = UnstructuredCSVLoader(
                uploaded_file_path, mode="elements", encoding="utf8"
            )
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(uploaded_file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(uploaded_file_path)
        elif file_extension == ".mp3":
            # Process audio files differently
            audio_file = genai.upload_file(path=uploaded_file_path)
            model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
            prompt = f"{prompt}"
            response = model.generate_content(
                [prompt, audio_file], safety_settings=safety_settings
            )
            summary = format_text(response.text)
            document_analyzed = True
            return AnalyzeDocumentResponse(
                meta={"status": "success", "code": 200}, summary=summary
            )

        # If no loader is set, raise an exception
        if loader is None:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_extension}"
            )

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
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        response = stuff_chain.invoke(docs)
        summary = format_text(response["output_text"])
        document_analyzed = True

        return AnalyzeDocumentResponse(
            meta={"status": "success", "code": 200}, summary=summary
        )

    except Exception as e:
        print(f"An error occurred during document analysis: {e}")  # Log the error
        raise HTTPException(
            status_code=500, detail="An error occurred during document analysis."
        )


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
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=api)

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
            model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
            latest_conversation = request.cookies.get("latest_question_response", "")
            prompt = (
                "Answer the question based on the speech: "
                + question
                + (
                    f" Latest conversation: {latest_conversation}"
                    if latest_conversation
                    else ""
                )
            )

            # Generate response based on audio input
            response = model.generate_content(
                [prompt, audio_file], safety_settings=safety_settings
            )
            current_response = response.text
            current_question = f"You asked: {question}"

            # Save the latest question and response to the session
            question_responses.append((current_question, current_response))

            # Use the summary generated from the MP3 content as text
            text = current_response

            # Set the Google API key
            os.environ["GOOGLE_API_KEY"] = api

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=8000, chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)

            # Generate embeddings for the chunks
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            document_search = FAISS.from_texts(
                [chunk.page_content for chunk in chunks], embeddings
            )

            if document_search:
                query_embedding = embeddings.embed_query(question)
                results = document_search.similarity_search_by_vector(
                    query_embedding, k=1
                )

                if results:
                    current_response = format_text(results[0].page_content)
                else:
                    current_response = "No matching document found in the database."
            else:
                current_response = "Vector database not initialized."

            return AskResponse(
                meta={"status": "success", "code": 200},
                question=question,
                result=current_response,
            )

        # If no loader is set, raise an exception
        if loader is None:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_extension}"
            )

        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(
            docs
        )  # Pass the list of Document objects

        # Generate embeddings for the chunks
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        document_search = FAISS.from_texts(
            [chunk.page_content for chunk in chunks], embeddings
        )

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f'{question} Answer the question based on the following:\n"{retrieved_texts}"\n:'
                + (
                    f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}"
                    if latest_conversation
                    else ""
                )
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            response_chain = llm_chain1.invoke({"text": retrieved_texts})
            summary1 = format_text(response_chain["text"])

            # Return the response
            return AskResponse(
                meta={"status": "success", "code": 200},
                question=question,
                result=summary1,
            )
        else:
            return AskResponse(
                meta={"status": "success", "code": 200},
                question=question,
                result="No relevant results found.",
            )

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
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    # Replace any remaining * with <br>
    text = text.replace("*", "<br>")
    return text


def clean_data(df):
    # Step 1: Clean currency-related columns
    for col in df.columns:
        if any(x in col.lower() for x in ["value", "price", "cost", "amount"]):
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .str.replace("$", "")
                    .str.replace("£", "")
                    .str.replace("€", "")
                    .replace("[^\d.-]", "", regex=True)
                    .astype(float)
                )

    # Step 2: Drop columns with more than 25% missing values
    null_percentage = df.isnull().sum() / len(df)
    columns_to_drop = null_percentage[null_percentage > 0.25].index
    df.drop(columns=columns_to_drop, inplace=True)

    # Step 3: Fill missing values for remaining columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if null_percentage[col] <= 0.25:
                if df[col].dtype in ["float64", "int64"]:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)

    # Step 4: Convert object-type columns to lowercase
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower()

    # Step 5: Drop columns with only one unique value
    unique_value_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=unique_value_columns, inplace=True)

    return df


def clean_data2(df):
    for col in df.columns:
        if (
            "value" in col
            or "price" in col
            or "cost" in col
            or "amount" in col
            or "Value" in col
            or "Price" in col
            or "Cost" in col
            or "Amount" in col
        ):
            if df[col].dtype == "object":
                df[col] = df[col].str.replace("$", "")
                df[col] = df[col].str.replace("£", "")
                df[col] = df[col].str.replace("€", "")
                df[col] = df[col].replace("[^\d.-]", "", regex=True).astype(float)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower()

    return df


def generate_plot(df, plot_path, plot_type):
    df = clean_data(df)
    excluded_words = ["name", "postal", "date", "phone", "address", "code", "id"]

    if plot_type == "countplot":
        cat_vars = [
            col
            for col in df.select_dtypes(include="object").columns
            if all(word not in col.lower() for word in excluded_words)
            and df[col].nunique() > 1
        ]

        for col in cat_vars:
            if df[col].nunique() > 10:
                top_categories = df[col].value_counts().index[:10]
                df[col] = df[col].apply(lambda x: x if x in top_categories else "Other")

        num_cols = len(cat_vars)
        num_rows = (num_cols + 1) // 2
        fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        for i, var in enumerate(cat_vars):
            category_counts = df[var].value_counts()
            top_values = category_counts.index[:10][::-1]
            filtered_df = df.copy()
            filtered_df[var] = pd.Categorical(
                filtered_df[var], categories=top_values, ordered=True
            )
            sns.countplot(x=var, data=filtered_df, order=top_values, ax=axs[i])
            axs[i].set_title(var)
            axs[i].tick_params(axis="x", rotation=30)

            total = len(filtered_df[var])
            for p in axs[i].patches:
                height = p.get_height()
                axs[i].annotate(
                    f"{height / total:.1%}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                )

            sample_size = filtered_df.shape[0]
            axs[i].annotate(
                f"Sample Size = {sample_size}",
                xy=(0.5, 0.9),
                xycoords="axes fraction",
                ha="center",
                va="center",
            )

        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

    elif plot_type == "histplot":
        num_vars = [
            col
            for col in df.select_dtypes(include=["int", "float"]).columns
            if all(word not in col.lower() for word in excluded_words)
        ]
        num_cols = len(num_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(
            nrows=num_rows, ncols=min(3, num_cols), figsize=(15, 5 * num_rows)
        )
        axs = axs.flatten()

        plot_index = 0

        for i, var in enumerate(num_vars):
            if len(df[var].unique()) == len(df):
                fig.delaxes(axs[plot_index])
            else:
                sns.histplot(df[var], ax=axs[plot_index], kde=True, stat="percent")
                axs[plot_index].set_title(var)
                axs[plot_index].set_xlabel("")

            sample_size = df.shape[0]
            axs[i].annotate(
                f"Sample Size = {sample_size}",
                xy=(0.5, 0.9),
                xycoords="axes fraction",
                ha="center",
                va="center",
            )

            plot_index += 1

        for i in range(plot_index, len(axs)):
            fig.delaxes(axs[i])

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


@app.post("/py/v1/result", response_model=AnalyzeDocument1Response)
async def result(
    api_key: str = Form(...),
    file: UploadFile = File(...),
    custom_question: str = Form(...),
):
    global uploaded_df

    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read the file depending on the extension
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension == ".csv":
        df = pd.read_csv(file_path, delimiter=",")
    elif file_extension == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file format")

    columns = df.columns.tolist()

    # Function to generate Gemini response based on the plot
    def generate_gemini_response(plot_path):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        img = Image.open(plot_path)
        response = model.generate_content(
            [custom_question + " Analyze the data insights from the chart.", img]
        )
        response.resolve()
        return response.text

    try:
        # Generate plots
        plot1_path = generate_plot(df, "static/plot1.png", "countplot")
        plot2_path = generate_plot(df, "static/plot2.png", "histplot")

        # Generate Gemini responses
        response1 = format_text(generate_gemini_response(plot1_path))
        response2 = format_text(generate_gemini_response(plot2_path))

        uploaded_df = df

        # Generate PDF
        def safe_encode(text):
            try:
                return text.encode("latin1", errors="replace").decode(
                    "latin1"
                )  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"

        pdf = FPDF()
        pdf.set_font("Arial", size=12)

        pdf.add_page()
        pdf.cell(200, 10, txt="Single Countplot Barchart", ln=True, align="C")
        pdf.image(plot1_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(
            200,
            10,
            txt="Single Countplot Barchart Google Gemini Response",
            ln=True,
            align="C",
        )
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response1))

        pdf.add_page()
        pdf.cell(200, 10, txt="Single Histoplot", ln=True, align="C")
        pdf.image(plot2_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(
            200, 10, txt="Single Histoplot Google Gemini Response", ln=True, align="C"
        )
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response2))

        pdf_file_path = os.path.join("static", "output.pdf")
        pdf.output(pdf_file_path)

        # Upload files to external endpoint
        url = "https://api.goarif.co/api/v1/Attachment/Upload/Paython"

        # Upload CSV
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(
                url, files={"file": (file_path, f, "text/csv")}
            )

        # Upload Plot1
        with open(plot1_path, "rb") as f:
            plot1_url = requests.post(
                url, files={"file": ("plot1.png", f, "image/png")}
            )

        # Upload Plot2
        with open(plot2_path, "rb") as f:
            plot2_url = requests.post(
                url, files={"file": ("plot2.png", f, "image/png")}
            )

        # Upload PDF
        with open(pdf_file_path, "rb") as f:
            pdf_url = requests.post(
                url, files={"file": ("output.pdf", f, "application/pdf")}
            )

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
            columns=", ".join(columns),
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
    # Changed to str to handle CSV string input
):
    global document_analyzed

    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Read the file content into a DataFrame
        file_extension = os.path.splitext(file.filename)[1]
        if file_extension == ".csv":
            df = pd.read_csv(file_path, delimiter=",")
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        df = clean_data(df)

        # Select the target variable and columns for analysis from the DataFrame
        if target_variable not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target variable '{target_variable}' not found in the dataset",
            )

        target_variable_data = df[target_variable]
        columns_for_analysis_data = df.drop(columns=[target_variable])

        # Concatenate target variable and columns for analysis into a single DataFrame
        df = pd.concat([target_variable_data, columns_for_analysis_data], axis=1)

        # Generate visualizations

        # Multiclass Barplot
        excluded_words = ["name", "postal", "date", "phone", "address", "id"]

        # Get the names of all columns with data type 'object' (categorical variables)
        cat_vars = [
            col
            for col in df.select_dtypes(include=["object"]).columns
            if all(word not in col.lower() for word in excluded_words)
        ]

        # Exclude the target variable from the list if it exists in cat_vars
        if target_variable in cat_vars:
            cat_vars.remove(target_variable)

        # Create a figure with subplots, but only include the required number of subplots
        num_cols = len(cat_vars)
        num_rows = (
            num_cols + 2
        ) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        # Create a count plot for each categorical variable
        for i, var in enumerate(cat_vars):
            top_categories = df[var].value_counts().nlargest(5).index
            filtered_df = df[
                df[var].notnull() & df[var].isin(top_categories)
            ]  # Exclude rows with NaN values in the variable

            # Replace less frequent categories with "Other" if there are more than 5 unique values
            if df[var].nunique() > 5:
                other_categories = df[var].value_counts().index[5:]
                filtered_df[var] = filtered_df[var].apply(
                    lambda x: x if x in top_categories else "Other"
                )

            sns.countplot(
                x=var, hue=target_variable, data=filtered_df, ax=axs[i], stat="percent"
            )
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)

            # Change y-axis label to represent percentage
            axs[i].set_ylabel("Percentage")

            # Annotate the subplot with sample size
            sample_size = df.shape[0]
            axs[i].annotate(
                f"Sample Size = {sample_size}",
                xy=(0.5, 0.9),
                xycoords="axes fraction",
                ha="center",
                va="center",
            )

        # Remove any remaining blank subplots
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

        plt.xticks(rotation=45)
        plt.tight_layout()
        plot3_path = "static/multiclass_barplot.png"
        plt.savefig(plot3_path)
        plt.close(fig)

        # Multiclass Histplot
        int_vars = df.select_dtypes(include=["int", "float"]).columns.tolist()
        int_vars = [col for col in int_vars if col != target_variable]

        # Create a figure with subplots
        num_cols = len(int_vars)
        num_rows = (
            num_cols + 2
        ) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        # Create a histogram for each integer variable with hue=target_variable
        for i, var in enumerate(int_vars):
            sns.histplot(
                data=df, x=var, hue=target_variable, kde=True, ax=axs[i], stat="percent"
            )
            axs[i].set_title(var)

            # Annotate the subplot with sample size
            sample_size = df.shape[0]
            axs[i].annotate(
                f"Sample Size = {sample_size}",
                xy=(0.5, 0.9),
                xycoords="axes fraction",
                ha="center",
                va="center",
            )

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
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response3 = format_text(
            model.generate_content([custom_question, img_barplot]).text
        )

        # Response for the histplot
        img_histplot = Image.open(plot4_path)
        response4 = format_text(
            model.generate_content([custom_question, img_histplot]).text
        )

        document_analyzed = True

        # Create a dictionary to store the outputs
        outputs = {
            "multiBarchart_visualization": plot3_path,
            "gemini_response3": response3,
            "multiHistoplot_visualization": plot4_path,
            "gemini_response4": response4,
        }

        # Save the dictionary as a JSON file
        with open("output1.json", "w") as outfile:
            json.dump(outputs, outfile)

        def safe_encode(text):
            try:
                return text.encode("latin1", errors="replace").decode(
                    "latin1"
                )  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"

        # Generate PDF with the results
        pdf = FPDF()
        pdf.set_font("Arial", size=12)

        # Add content to the PDF
        # Multiclass Countplot Barchart and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Countplot Barchart", ln=True, align="C")
        pdf.image(plot3_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(
            200,
            10,
            txt="Multiclass Countplot Barchart Google Gemini Response",
            ln=True,
            align="C",
        )
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response3))

        # Multiclass Histplot and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Histplot", ln=True, align="C")
        pdf.image(plot4_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(
            200,
            10,
            txt="Multiclass Histplot Google Gemini Response",
            ln=True,
            align="C",
        )
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response4))

        pdf_output_path = "static/analysis_report_complete.pdf"
        pdf.output(pdf_output_path)
        pdf_file_path = pdf_output_path.replace("\\", "/")

        # Upload files to external endpoint
        url = "https://api.goarif.co/api/v1/Attachment/Upload/Paython"

        # Upload CSV
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(
                url, files={"file": (file_path, f, "text/csv")}
            )

        # Upload Plot1
        with open(plot3_path, "rb") as f:
            plot3_url = requests.post(
                url, files={"file": ("multiclass_barplot.png", f, "image/png")}
            )

        # Upload Plot2
        with open(plot4_path, "rb") as f:
            plot4_url = requests.post(
                url, files={"file": ("multiclass_histplot.png", f, "image/png")}
            )

        # Upload PDF
        with open(pdf_file_path, "rb") as f:
            pdf_url = requests.post(
                url,
                files={"file": ("analysis_report_complete.pdf", f, "application/pdf")},
            )

        return MulticlassResponse(
            meta={"status": "success", "code": 200},
            plot3_path=plot3_url.text,
            plot4_path=plot4_url.text,
            response3=response3,
            response4=response4,
            pdf_file_path=pdf_url.text,
            file_path=uploaded_file_url.text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Route for answering questions
@app.post("/py/v1/ask1", response_model=AskResponse1)
async def ask_question(
    request: Request,
    api_key: str = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...),
):
    global uploaded_file_path, document_analyzed, summary, api, llm

    loader = None

    try:
        # Initialize the LLM model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", google_api_key=api_key
        )

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
            raise HTTPException(
                status_code=500, detail=f"Error loading document: {str(e)}"
            )

        # Combine document text
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api_key

        # Initialize embeddings and create FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(
            docs
        )  # Pass the list of Document objects
        document_search = FAISS.from_texts(
            [chunk.page_content for chunk in chunks], embeddings
        )

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f'{question} Answer the question based on the following:\n"{text}"\n:'
                + (
                    f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}"
                    if latest_conversation
                    else ""
                )
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            try:
                response_chain = llm_chain1.invoke({"text": text})
                summary1 = response_chain["text"]
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error invoking LLMChain: {str(e)}"
                )

            # Generate embeddings for the summary
            try:
                summary_embedding = embeddings.embed_query(summary1)
                document_search = FAISS.from_texts([summary1], embeddings)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error generating embeddings: {str(e)}"
                )

            # Perform a search on the FAISS vector database
            try:
                if document_search:
                    query_embedding = embeddings.embed_query(question)
                    results = document_search.similarity_search_by_vector(
                        query_embedding, k=1
                    )

                    if results:
                        current_response = format_text(results[0].page_content)
                    else:
                        current_response = "No matching document found in the database."
                else:
                    current_response = "Vector database not initialized."
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error during similarity search: {str(e)}"
                )
        else:
            current_response = "No relevant results found."

        # Append the question and response from FAISS search
        current_question = f"You asked: {question}"
        question_responses.append((current_question, current_response))

        # Save all results to output_summary.json
        save_to_json(question_responses)

        return AskResponse1(
            meta={"status": "success", "code": 200},
            question=question,
            result=current_response,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


def save_to_json(question_responses):
    outputs = {"question_responses": question_responses}
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
more_stopword = ["yg", "yang", "aku", "gw", "gua", "gue"]
data = stop_factory.get_stop_words() + more_stopword

# Define hyperlink pattern for removal
hyperlink_pattern = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
number_pattern = re.compile(r"\b\d+\b")

emoticon_pattern = re.compile(
    "("
    "\ud83c[\udf00-\udfff]|"
    "\ud83d[\udc00-\ude4f\ude80-\udeff]|"
    "[\u2600-\u26ff\u2700-\u27bf])+",
    re.UNICODE,
)


@app.post("/py/v1/process", response_model=GetColumn)
async def process_file(request: Request, file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load DataFrame based on file type
    file_extension = os.path.splitext(file.filename)[1]
    try:
        if file_extension == ".csv":
            # Load CSV file into DataFrame
            df = pd.read_csv(file_path, delimiter=",")
        elif file_extension in [".xls", ".xlsx"]:
            # Load Excel file into DataFrame
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        # Filter columns based on the condition
        valid_columns = []
        for column in df.columns:
            if any(df[column].dropna().astype(str).apply(len) > 15):
                valid_columns.append(column)

        # Upload CSV
        url = "https://api.goarif.co/api/v1/Attachment/Upload/Paython"
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(
                url, files={"file": (file_path, f, "text/csv")}
            )

        return GetColumn(
            meta={"status": "success", "code": 200},
            columns=", ".join(valid_columns),  # Use filtered columns
            file_path=uploaded_file_url.text,  # Return the external file URL
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
    custom_question: str = Form(""),
):
    global df
    # Read the uploaded CSV file

    from huggingface_hub import login

    login(hf_token)

    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load DataFrame based on file type
    file_extension = os.path.splitext(file.filename)[1]
    try:
        if file_extension == ".csv":
            df = pd.read_csv(file_path, delimiter=",")
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        if target_variable not in df.columns:
            return "Selected target variable does not exist in the dataset."

        add_stopwords = [
            "the",
            "of",
            "is",
            "a",
            "in",
            "https",
            "yg",
            "gua",
            "gue",
            "lo",
            "lu",
            "gw",
        ]
        custom_stopword_list = [word.strip() for word in custom_stopwords.split(",")]
        all_stopwords = data + custom_stopword_list + add_stopwords

        # Remove hyperlinks, emoticons, numbers, and stopwords
        hyperlink_pattern = r"https?://\S+|www\.\S+"
        emoticon_pattern = r"[:;=X][oO\-]?[D\)\]\(\]/\\OpP]"
        number_pattern = r"\b\d+\b"

        df[target_variable] = df[target_variable].astype(str)
        df["cleaned_text"] = df[target_variable].str.replace(
            hyperlink_pattern, "", regex=True
        )
        df["cleaned_text"] = df["cleaned_text"].str.replace(
            emoticon_pattern, "", regex=True
        )
        df["cleaned_text"] = df["cleaned_text"].str.replace(
            number_pattern, "", regex=True
        )
        for stopword in all_stopwords:
            df["cleaned_text"] = df["cleaned_text"].apply(
                lambda x: " ".join(
                    [word for word in x.split() if word.lower() != stopword]
                )
            )

        # Perform stopwords removal
        # df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(
        # [stemmer.stem(word) for word in stop_factory.create_stop_word_remover().remove(x).split()
        # if word.lower() not in all_stopwords]
        # ))

        # Perform Sentiment Analysis
        pretrained = "mdhugol/indonesia-bert-sentiment-classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained, token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained, token=hf_token)
        sentiment_analysis = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )
        label_index = {
            "LABEL_0": "positive",
            "LABEL_1": "neutral",
            "LABEL_2": "negative",
        }

        def analyze_sentiment(text):
            result = sentiment_analysis(text)
            label = label_index[result[0]["label"]]
            score = result[0]["score"]
            return pd.Series({"sentiment_label": label, "sentiment_score": score})

        df[["sentiment_label", "sentiment_score"]] = df["cleaned_text"].apply(
            analyze_sentiment
        )

        # Count the occurrences of each sentiment label
        sentiment_counts = df["sentiment_label"].value_counts()

        # Plot a bar chart using seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis"
        )
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment Label")
        plt.ylabel("Count")
        sentiment_plot_path = "static/sentiment_distribution.png"
        plt.savefig(sentiment_plot_path)

        # ApexCharts configuration
        apex_chart_config = {
            "chart": {"type": "bar"},
            "plotOptions": {"bar": {"horizontal": True}},
            "series": [
                {
                    "data": [
                        {"x": label, "y": count}
                        for label, count in sentiment_counts.items()
                    ]
                }
            ],
        }

        # Filter the DataFrame to include only the desired columns
        filtered_df = df[[target_variable, "sentiment_label"]]

        # Generate the HTML table for the filtered DataFrame
        analysis_results = filtered_df.to_html(classes="data")

        # Convert the DataFrame to a list of dictionaries
        # table_data_list = filtered_df.to_dict('records')

        # Optionally, save the filtered DataFrame to a CSV file
        filtered_df.to_csv("sentiment_analysis_results.csv", index=False)

        # Concatenate Cleaned text
        positive_text = " ".join(
            df[df["sentiment_label"] == "positive"]["cleaned_text"]
        )
        negative_text = " ".join(
            df[df["sentiment_label"] == "negative"]["cleaned_text"]
        )
        neutral_text = " ".join(df[df["sentiment_label"] == "neutral"]["cleaned_text"])

        # Create WordCloud Positive
        wordcloud = WordCloud(
            min_font_size=3,
            max_words=200,
            width=800,
            height=400,
            colormap="Set2",
            background_color="white",
        ).generate(positive_text)

        wordcloud_positive = "static/wordcloud_positive.png"
        wordcloud.to_file(wordcloud_positive)

        # Use Google Gemini API to generate content based on the uploaded image
        img = PIL.Image.open(wordcloud_positive)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        try:
            response = model.generate_content(
                [
                    custom_question
                    + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.",
                    img,
                ]
            )
            response.resolve()
            gemini_response_pos = format_text(response.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_pos = "Error: Failed to generate content with Gemini API."

        # Create WordCloud Neutral
        wordcloud = WordCloud(
            min_font_size=3,
            max_words=200,
            width=800,
            height=400,
            colormap="Set2",
            background_color="white",
        ).generate(neutral_text)

        wordcloud_neutral = "static/wordcloud_neutral.png"
        wordcloud.to_file(wordcloud_neutral)

        img = PIL.Image.open(wordcloud_neutral)
        try:
            response = model.generate_content(
                [
                    custom_question
                    + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.",
                    img,
                ]
            )
            response.resolve()
            gemini_response_neu = format_text(response.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neu = "Error: Failed to generate content with Gemini API."

        # Create WordCloud Negative
        wordcloud = WordCloud(
            min_font_size=3,
            max_words=200,
            width=800,
            height=400,
            colormap="Set2",
            background_color="white",
        ).generate(negative_text)

        wordcloud_negative = "static/wordcloud_negative.png"
        wordcloud.to_file(wordcloud_negative)

        img = PIL.Image.open(wordcloud_negative)
        try:
            response = model.generate_content(
                [
                    custom_question
                    + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.",
                    img,
                ]
            )
            response.resolve()
            gemini_response_neg = format_text(response.text)
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neg = "Error: Failed to generate content with Gemini API."

        from PIL import Image

        # Combine WordClouds (Positive, Neutral, Negative)
        wordclouds = [
            sentiment_plot_path,
            wordcloud_positive,
            wordcloud_neutral,
            wordcloud_negative,
        ]

        # Open the three wordcloud images
        images = [Image.open(wc) for wc in wordclouds]

        # Assuming all wordclouds are the same size, we can place them side by side
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create a new image with combined width and max height
        combined_image = Image.new("RGB", (total_width, max_height))

        # Paste each image into the new combined image
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save the combined image to the static folder
        combined_wordcloud_path = "static/wordcloud_combined.png"
        combined_image.save(combined_wordcloud_path)

        def generate_gemini_response(plot_path):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            img = Image.open(plot_path)
            response = model.generate_content(
                [
                    custom_question
                    + " As a marketing consultant, I want to analyze consumer insights from the sentiment word clouds (positive, neutral, and negative) and the market context. Please summarize your explanation and findings in one concise paragraph and one another paragraph for business insight and reccomendation to help me formulate actionable strategies.",
                    img,
                ]
            )
            response.resolve()
            return response.text

        response_result = format_text(generate_gemini_response(combined_wordcloud_path))

        document_analyzed = True

        # Function to handle encoding to latin1
        def safe_encode(text):
            try:
                return text.encode("latin1", errors="replace").decode(
                    "latin1"
                )  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"

        # Generate PDF with the results
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align="C")

        # Sentiment Distribution Plot
        pdf.image(sentiment_plot_path, x=10, y=30, w=190)
        pdf.ln(100)

        # Positive WordCloud and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive WordCloud", ln=True, align="C")
        pdf.image(wordcloud_positive, x=10, y=30, w=190)
        pdf.add_page()

        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_pos))

        # Negative WordCloud and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative WordCloud", ln=True, align="C")
        pdf.image(wordcloud_negative, x=10, y=30, w=190)
        pdf.add_page()

        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neg))

        pdf_file_path = os.path.join("static", "sentiment.pdf")
        pdf.output(pdf_file_path)

        pdf_file_path = os.path.join("static", "sentiment.pdf")
        pdf_file_path = pdf_file_path.replace("\\", "/")

        # Upload files to external endpoint
        url = "https://api.goarif.co/api/v1/Attachment/Upload/Paython"

        # Upload CSV
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(
                url, files={"file": (file_path, f, "text/csv")}
            )

        # Upload Plot1
        with open(sentiment_plot_path, "rb") as f:
            sentiment_url = requests.post(
                url, files={"file": ("sentiment_distribution.png", f, "image/png")}
            )

        # Upload wordcloud
        with open(wordcloud_positive, "rb") as f:
            wordcloud_positive_url = requests.post(
                url, files={"file": ("wordcloud_positive.png", f, "image/png")}
            )

        # Upload Plot2
        with open(wordcloud_negative, "rb") as f:
            wordcloud_negative_url = requests.post(
                url, files={"file": ("wordcloud_negative.png", f, "image/png")}
            )

        # Upload PDF
        with open(pdf_file_path, "rb") as f:
            pdf_url = requests.post(
                url, files={"file": ("sentiment.pdf", f, "application/pdf")}
            )

        return AnalyzeDocumentResponse2(
            meta={"status": "success", "code": 200},
            sentiment_plot_path=sentiment_url.text,
            analysis_results=analysis_results,
            wordcloud_positive=wordcloud_positive_url.text,
            gemini_response_pos=gemini_response_pos,
            wordcloud_negative=wordcloud_negative_url.text,
            gemini_response_neg=gemini_response_neg,
            response_result=response_result,
            pdf_file_path=pdf_url.text,
            file_path=uploaded_file_url.text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Route for answering questions
@app.post("/py/v1/ask2", response_model=AskResponse2)
async def ask_question(
    request: Request,
    api_key: str = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...),
):
    global uploaded_file_path, document_analyzed, summary, api, llm

    loader = None

    try:
        # Initialize the LLM model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", google_api_key=api_key
        )

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
            raise HTTPException(
                status_code=500, detail=f"Error loading document: {str(e)}"
            )

        # Combine document text
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api_key

        # Initialize embeddings and create FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(
            docs
        )  # Pass the list of Document objects
        document_search = FAISS.from_texts(
            [chunk.page_content for chunk in chunks], embeddings
        )

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f'{question} Answer the question based on the following:\n"{text}"\n:'
                + (
                    f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}"
                    if latest_conversation
                    else ""
                )
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            try:
                response_chain = llm_chain1.invoke({"text": text})
                summary1 = response_chain["text"]
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error invoking LLMChain: {str(e)}"
                )

            # Generate embeddings for the summary
            try:
                summary_embedding = embeddings.embed_query(summary1)
                document_search = FAISS.from_texts([summary1], embeddings)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error generating embeddings: {str(e)}"
                )

            # Perform a search on the FAISS vector database
            try:
                if document_search:
                    query_embedding = embeddings.embed_query(question)
                    results = document_search.similarity_search_by_vector(
                        query_embedding, k=1
                    )

                    if results:
                        current_response = format_text(results[0].page_content)
                    else:
                        current_response = "No matching document found in the database."
                else:
                    current_response = "Vector database not initialized."
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error during similarity search: {str(e)}"
                )
        else:
            current_response = "No relevant results found."

        # Append the question and response from FAISS search
        current_question = f"You asked: {question}"
        question_responses.append((current_question, current_response))

        # Save all results to output_summary.json
        save_to_json(question_responses)

        return AskResponse2(
            meta={"status": "success", "code": 200},
            question=question,
            result=current_response,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


def save_to_json(question_responses):
    outputs = {"question_responses": question_responses}
    with open("output_summary.json", "w") as outfile:
        json.dump(outputs, outfile)


@app.get("/download_pdf")
async def download_pdf():
    pdf_output_path = "static/analysis_report.pdf"
    return FileResponse(
        pdf_output_path, filename="analysis_report.pdf", media_type="application/pdf"
    )


### CLUSTERING ANALYSIS ----------------------------------------------------------------
@app.post("/py/v1/getcolumn", response_model=GetResult)
async def process_file(request: Request, file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load DataFrame based on file type
    file_extension = os.path.splitext(file.filename)[1]
    try:
        if file_extension == ".csv":
            df = pd.read_csv(file_path, delimiter=",")
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        # Get columns of the DataFrame
        columns = df.columns.tolist()

        # Upload CSV
        url = "https://api.goarif.co/api/v1/Attachment/Upload/Paython"
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(
                url, files={"file": (file_path, f, "text/csv")}
            )

        return GetResult(
            meta={"status": "success", "code": 200},
            columns=", ".join(columns),
            file_path=uploaded_file_url.text,  # Return the external file URL
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/py/v1/cluster", response_model=AnalyzeDocumentResponse3)
async def result(
    request: Request,
    question: str = Form(...),
    api_key: str = Form(...),
    file: UploadFile = File(...),
    clustering_columns: str = Form(...),
):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Save the uploaded file

        # Load the file into a DataFrame
        file_extension = os.path.splitext(file.filename)[1]
        if file_extension == ".csv":
            df = pd.read_csv(file_path, delimiter=",")
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        print("Columns in the uploaded file:", df.columns.tolist())

        # Parse clustering_columns into a list
        clustering_columns_list = [col.strip() for col in clustering_columns.split(",")]

        print("Clustering Columns Provided:", clustering_columns_list)
        missing_columns = [
            col for col in clustering_columns_list if col not in df.columns
        ]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"The following columns are missing in the uploaded file: {', '.join(missing_columns)}",
            )

        # Create a copy of selected data for transformation
        selected_data = df[clustering_columns_list].copy()
        print("Selected Data for Clustering (before encoding):", selected_data.head())

        # Store original values for categorical columns
        original_categorical_values = {}

        # Apply Label Encoding only for object columns
        from sklearn import preprocessing

        label_encoders = {}
        for column in selected_data.columns:
            if selected_data[column].dtype == "object":
                print(f"Encoding column: {column}")
                label_encoders[column] = preprocessing.LabelEncoder()
                # Store original values before encoding
                original_categorical_values[column] = selected_data[column].copy()
                try:
                    selected_data[column] = label_encoders[column].fit_transform(
                        selected_data[column].astype(str)
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Error encoding column {column}: {e}"
                    )

        print("Transformed Data for Clustering:", selected_data.head())

        # Apply DBSCAN Clustering
        dbscan = DBSCAN(eps=0.7, min_samples=8)
        cluster_labels = dbscan.fit_predict(selected_data)

        print("Cluster Labels:", cluster_labels)

        # Add cluster labels to the original DataFrame
        df["cluster"] = cluster_labels

        print("Final DataFrame with Clusters:", df.head())

        # Calculate range values for each cluster and selected columns
        ranges = []
        unique_clusters = sorted(df["cluster"].unique())

        for cluster in unique_clusters:
            if cluster == -1:
                ranges.append(
                    {
                        "Cluster": "Noise",
                        clustering_columns_list[0]: "N/A",
                        clustering_columns_list[1]: "N/A",
                    }
                )
            else:
                cluster_data = df[df["cluster"] == cluster]

                def get_cluster_value(column):
                    if column in original_categorical_values:
                        # For categorical columns, use mode of original values
                        return cluster_data[column].mode().iloc[0]
                    else:
                        # For numerical columns, use mean
                        return f"{cluster_data[column].mean():.2f}"

                ranges.append(
                    {
                        "Cluster": cluster + 1,
                        clustering_columns_list[0]: get_cluster_value(
                            clustering_columns_list[0]
                        ),
                        clustering_columns_list[1]: get_cluster_value(
                            clustering_columns_list[1]
                        ),
                    }
                )

        # Create a DataFrame for ranges
        range_df = pd.DataFrame(ranges)
        range_df = range_df[range_df["Cluster"] != "Noise"]

        range_table_html = range_df.to_html(
            index=False, classes="table table-striped", border=0
        )
        # Convert the DataFrame to a list of dictionaries
        # table_data_list = range_df.to_dict('records')

        # Save the DataFrame as a CSV file
        result_csv_path = "result.csv"
        range_df.to_csv(result_csv_path, index=False)

        # Step 1: Load the DataFrame from result.csv
        df1 = pd.read_csv("result.csv")

        # Apply Label Encoding only for object columns in df1
        for column in df1.columns:
            if df1[column].dtype == "object":
                print(f"Encoding column: {column}")
                if column in label_encoders:
                    df1[column] = label_encoders[column].transform(
                        df1[column].astype(str)
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Label encoder not found for column {column}",
                    )

        print("Transformed df1 for Clustering:", df1.head())

        # Step 2: Encode all object columns in df1 using OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse_output=False)
        encoded_features = onehot_encoder.fit_transform(
            df1[clustering_columns_list].select_dtypes(include=["object"])
        )
        encoded_columns = onehot_encoder.get_feature_names_out(
            df1[clustering_columns_list].select_dtypes(include=["object"]).columns
        )
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)

        # Step 3: Combine numerical features with encoded features
        numerical_features = df1[clustering_columns_list].select_dtypes(
            exclude=["object"]
        )
        features = pd.concat([numerical_features, encoded_df], axis=1)

        # Step 4: Compute cosine similarity between users based on features
        user_similarity = cosine_similarity(features)

        # Step 5: Create a DataFrame to display the similarity between users
        user_similarity_df = pd.DataFrame(
            user_similarity, index=df1["Cluster"], columns=df1["Cluster"]
        )

        def generate_product_recommendation_table(
            ranged_df, user_similarity_df, max_groups=4
        ):
            recommendations = {
                1: "Paket JITU 1 : Internet dan TV 100 Mbps",
                2: "Paket Dynamic Telkomsel One : Kecepatan Wi-fi 300 Mbps, Paket Data 15GB",
                3: "Paket Add-On Netflix : Netflix + Vidio & WeTV",
                4: "Paket Prime Video : Konten Amazon Original",
            }

            recommendation_data = []
            groupings = {}
            similarity_threshold = 0.8

            for cluster in ranged_df["Cluster"]:
                similar_clusters = (
                    user_similarity_df[cluster].sort_values(ascending=False).index[1:]
                )
                similar_clusters = [
                    cl
                    for cl in similar_clusters
                    if user_similarity_df[cluster][cl] > similarity_threshold
                ]

                if len(similar_clusters) > 0:
                    group_key = tuple(sorted(similar_clusters))
                    if group_key not in groupings:
                        groupings[group_key] = []
                    groupings[group_key].append(cluster)

            grouped_clusters = list(groupings.values())
            while len(grouped_clusters) > max_groups:
                grouped_clusters = sorted(grouped_clusters, key=len)
                merged_group = grouped_clusters[0] + grouped_clusters[1]
                grouped_clusters = grouped_clusters[2:]
                grouped_clusters.append(merged_group)

            for group in grouped_clusters:
                representative_cluster = group[0]
                recommendation = recommendations.get(
                    representative_cluster, recommendations.get(1)
                )

                recommendation_data.append(
                    {
                        "Group": f"Group {len(recommendation_data) + 1}",
                        "Cluster IDs": ", ".join(map(str, group)),
                        "Recommendation": recommendation,
                    }
                )

            recommendation_df = pd.DataFrame(recommendation_data)
            return recommendation_df

        recommendation_df = generate_product_recommendation_table(
            df1, user_similarity_df, max_groups=4
        )

        recommendation_table_html = recommendation_df.to_html(
            index=False, classes="table table-striped", border=0
        )

        # Convert the DataFrame to a list of dictionaries
        # table_data_list1 = recommendation_df.to_dict('records')

        result_csv_path = "result1.csv"
        recommendation_df.to_csv(result_csv_path, index=False)

        df_result = pd.read_csv("result.csv")
        df_result1 = pd.read_csv("result1.csv")

        combined_df = pd.concat([df_result, df_result1], ignore_index=True)

        result_csv_path1 = "combined_result.csv"
        combined_df.to_csv(result_csv_path1, index=False)

        api = api_key
        genai.configure(api_key=api)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", google_api_key=api
        )

        # Load and process the CSV file
        loader = UnstructuredCSVLoader(result_csv_path1, mode="elements")
        docs = loader.load()

        # Convert documents to text
        context_text = "\n".join([doc.page_content for doc in docs])

        # Create the template
        template1 = """
        Based on the following retrieved context:
        {text}

        You are a skilled Data Analyst. Analyze the following dataset with a focus on providing a clear summary, actionable insights, and group characteristics. Specifically, consider the following aspects:
        1. Summarize the main characteristics of each group based on the average values and distribution of features for the clusters within each group.
        2. Identify any notable patterns or trends between the groups and explain what differentiates them.
        3. Suggest potential business strategies or actions based on these group characteristics (e.g., marketing strategies, product offerings, customer segmentation).
        4. Provide recommendations for how each group might be targeted or engaged differently to maximize value.
        5. Suggest a descriptive and intuitive name for each group.
        6. Group the clusters into meaningful categories.

        {question}
        """

        # Create prompt template with correct input variables
        prompt_template = PromptTemplate(
            input_variables=["text", "question"], template=template1
        )

        # Create and run the chain with both required inputs
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        response = llm_chain.invoke({"text": context_text, "question": question})

        # Extract the answer from the response
        summary = format_text(
            response["text"] if isinstance(response, dict) else response
        )

        url = "https://api.goarif.co/api/v1/Attachment/Upload/Paython"
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(
                url, files={"file": (file_path, f, "text/csv")}
            )

        return AnalyzeDocumentResponse3(
            meta={"status": "success", "code": 200},
            file_path=uploaded_file_url.text,
            table_data=range_table_html,
            recommendation_table_html=recommendation_table_html,
            summary=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


### PREDICTIVE ANALYSIS ----------------------------------------------------------------


@app.post("/py/v1/getcolumn1", response_model=GetResult1)
async def process_file(request: Request, file: UploadFile = File(...)):
    try:
        if file.filename == "":
            raise HTTPException(status_code=400, detail="No file selected")

        uploaded_filename = secure_filename(file.filename)
        file_path = os.path.join("static", uploaded_filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load DataFrame based on file type
        file_extension = os.path.splitext(file.filename)[1]
        if file_extension == ".csv":
            df = pd.read_csv(file_path, delimiter=",")
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        # Filter columns where unique value count is less than 5
        filtered_columns = [column for column in df.columns if df[column].nunique() < 5]

        # Upload file to external API
        url = "https://api.goarif.co/api/v1/Attachment/Upload/Paython"
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(
                url, files={"file": (file_path, f, "text/csv")}
            )

        return GetResult1(
            meta={"status": "success", "code": 200},
            columns=", ".join(filtered_columns),  # Return only filtered columns
            file_path=uploaded_file_url.text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/py/v1/predictive", response_model=AnalyzeDocumentResponse4)
async def result(
    request: Request,
    question: str = Form(...),
    api_key: str = Form(...),
    file: UploadFile = File(...),
    target_variable: str = Form(...),
):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Save the uploaded file

        file_extension = os.path.splitext(file.filename)[1]
        if file_extension == ".csv":
            df = pd.read_csv(file_path, delimiter=",")
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        # Data Cleansing Steps
        original_df = df.copy()

        # 1. Drop columns containing 'id' in their name (case insensitive)
        id_columns = [col for col in df.columns if "id" in col.lower()]
        df = df.drop(columns=id_columns)

        # 2. Drop categorical columns with more than 10 unique values
        categorical_columns = df.select_dtypes(include=["object"]).columns
        high_cardinality_cols = [
            col
            for col in categorical_columns
            if df[col].nunique() > 10 and col != target_variable
        ]
        df = df.drop(columns=high_cardinality_cols)

        # 3. Handle missing values
        df = df.fillna(df.mode().iloc[0])

        # Save original label mappings for object data types
        label_encoders = {}
        label_mappings = {}
        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            label_mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))

        # Check if target_variable exists in the DataFrame
        if target_variable not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target variable '{target_variable}' not found in dataset.",
            )

        # Splitting data into train and test sets
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Hyperparameter tuning for Random Forest
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", None],
        }

        rf = RandomForestClassifier(random_state=0)
        grid_search = GridSearchCV(
            estimator=rf, param_grid=param_grid, cv=5, scoring="accuracy"
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predict probabilities for entire cleaned dataset
        predictions_proba = best_model.predict_proba(X)
        predictions_df = pd.DataFrame(
            predictions_proba, columns=[f"Proba_{cls}" for cls in best_model.classes_]
        )

        # Add original values back
        for col, le in label_encoders.items():
            if col in original_df.columns:
                predictions_df[col] = original_df[col]

        # Add target variable
        predictions_df[target_variable] = original_df[target_variable]

        # Convert the DataFrame to HTML
        table_data = predictions_df.to_html(
            index=False, classes="table table-striped", border=0
        )

        # Convert the DataFrame to a list of dictionaries
        # table_data_list = predictions_df.to_dict('records')

        # import json
        # table_data_json = json.dumps(table_data_list)

        print("Data successfully written to output.json")

        # Save the DataFrame as a CSV file
        result_csv_path = "result.csv"
        predictions_df.to_csv(result_csv_path, index=False)

        api = api_key
        genai.configure(api_key=api)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", google_api_key=api
        )

        # Load the result CSV data for analysis
        loader = UnstructuredCSVLoader(result_csv_path, mode="elements")
        docs = loader.load()

        template1 = f"""
        Based on the following retrieved context:
        {{text}}
        
        You are a skilled Data Analyst tasked with performing a customer churn analysis. Please focus on the following aspects:
        
        1. **Churn Drivers**:
           - Identify the main factors contributing to customer churn based on the dataset.
           - Explain how these factors influence customer behavior and churn probability.
        
        2. **Churn Prediction Insights**:
           - Analyze patterns and trends among customers who are likely to churn and those who are not.
           - Highlight the key differences between these groups in terms of their characteristics and behaviors.
        
        3. **Actionable Strategies**:
           - Suggest potential strategies to reduce customer churn, such as targeted retention campaigns, personalized offers, or service improvements.
           - Provide specific actions for addressing the primary drivers of churn identified in the analysis.
        
        4. **Customer Segmentation**:
           - Based on the analysis, segment customers into groups such as 'High Risk of Churn', 'Moderate Risk', and 'Low Risk'.
           - For each segment:
             - Provide a descriptive name.
             - Explain the key characteristics of customers in this segment.
        
        5. **Engagement Recommendations**:
           - Propose targeted engagement strategies for each customer segment to maximize retention and lifetime value.
        
        6. **Business Impact**:
           - Quantify the potential impact of the suggested strategies on customer retention and overall business performance.
        
        The goal is to derive actionable insights that can directly inform retention strategies and improve customer satisfaction.
        
        {question}
        """

        prompt_template = PromptTemplate.from_template(template1)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        response = stuff_chain.invoke(docs)
        summary = format_text(response["output_text"])

        # Upload files to external endpoint
        url = "https://api.goarif.co/api/v1/Attachment/Upload/Paython"

        # Upload CSV
        with open(file_path, "rb") as f:
            uploaded_file_url = requests.post(
                url, files={"file": (file_path, f, "text/csv")}
            )

        return AnalyzeDocumentResponse4(
            meta={"status": "success", "code": 200},
            file_path=uploaded_file_url.text,
            table_data=table_data,
            summary=summary,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9000)
