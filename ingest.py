import glob
import re
import pandas as pd

# from langchain.document_loaders import CSVLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader, PyPDFDirectoryLoader
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.vectorstores.faiss import FAISS

from langchain_community.document_loaders import CSVLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader, PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def create_vector_db():
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    question_answer_vectorstore = load_question_answer(embedding)
    pdf_vectorstore = load_pdf(embedding)

    for doc in glob.glob(f"{DATA_PATH}/*.docx"):
        docx_vectorstore = load_docx(embedding, doc)
        question_answer_vectorstore.merge_from(docx_vectorstore)

    question_answer_vectorstore.merge_from(pdf_vectorstore)

    question_answer_vectorstore.save_local(DB_FAISS_PATH)


def load_pdf(embedding):
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(texts, embedding)
    return vectorstore


def load_docx(embedding, doc_path):
    loader = Docx2txtLoader(doc_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(texts, embedding)
    return vectorstore


def load_genes_csv(embedding, filename):
    """Function to load Gene database"""

    loader = CSVLoader(f"{DATA_PATH}/{filename}", encoding="utf-8", csv_args={
        'delimiter': ','})
    data = loader.load()

    vectorstore = FAISS.from_documents(data, embedding)
    return vectorstore


def load_question_answer(embedding):
    """Function to load all required documents in a DataFrame"""
    # Load FAQ Excel
    faq = pd.read_excel(f"{DATA_PATH}/faq.xlsx", header=2)
    faq.ffill(inplace=True)

    # Load Questions from PDF
    pdf_question_answer = []
    with open("data/slides_questions.txt", "r") as file:
        for line in file:
            result = re.search(r"(.*\?)(.*)", line)
            if result:
                pdf_question_answer.append({"Question": result[1], "Nalagenetics (Standardized)": result[2]})
    pdf_dataframe = pd.DataFrame(pdf_question_answer)

    # Combine both FAQ and Slides questions into a single dataframe
    faq = pd.concat([faq, pdf_dataframe], ignore_index=True)

    vectorstore = FAISS.from_texts(
        [f"Q:{row[0]} A:{row[1]}" for row in faq[["Question", "Nalagenetics (Standardized)"]].to_numpy()],
        embedding=embedding
    )

    return vectorstore


if __name__ == "__main__":
    create_vector_db()