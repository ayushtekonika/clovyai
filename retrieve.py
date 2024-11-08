import os
import logging
import time
from uuid import uuid4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the Mistral API key from environment variables
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY is not set in the environment.")

# Initialize embedding model
logging.info("Initializing embedding model")
embeddings = MistralAIEmbeddings(model="mistral-embed")

def get_file_paths(directory):
    """Get all .pdf and .docx file paths from a given directory and its subdirectories."""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx')):
                file_paths.append(os.path.join(root, file))
    return file_paths

def ingest_files(directory):
    """Ingest all PDF and DOCX files from a directory, process text, split, and embed in vector store."""
    file_paths = get_file_paths(directory)
    if not file_paths:
        logging.warning("No PDF or DOCX files found in the directory.")
        return None

    all_splits = []
    for path in file_paths:
        try:
            logging.info(f'Starting processing of the document {path}...')
            # Load documents based on the file extension
            if path.lower().endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.lower().endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(path)
            else:
                logging.warning(f"Unsupported file type: {path}")
                continue

            docs = loader.load()

            # Split documents
            logging.info(f"Splitting docs from {path}")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            continue

    if not all_splits:
        logging.warning("No documents were split or added for embedding.")
        return None

    # Embed documents in vector store
    logging.info("Embedding docs in vector store")
    try:
        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        collection_name = "orion-fake-company-collection"
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db_temp",
        )
        # for doc, doc_id in zip(all_splits, uuids):
        #     print("doc", doc)
        # if doc:
        vectorstore.add_documents(documents=all_splits, ids=uuids)
        # logging.info(f"Document {doc_id} embedded successfully. Waiting for 2 seconds before the next document.")
        logging.info("Embedding completed successfully for all documents")
    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        raise

    return vectorstore


def retrieve_documents(query, top_k=5):
    """Load the existing vectorstore and retrieve top_k relevant documents based on the query."""
    try:
        collection_name = "orion-fake-company-collection"
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db_temp",
        )
        logging.info(f"Retrieving top {top_k} documents for query: {query}")
        return vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        logging.error(f"Error during document retrieval: {e}")
        raise

def retrieve_as_retriever():
    """Load the existing vectorstore and retrieve top_k relevant documents based on the query."""
    try:
        collection_name = "orion-fake-company-collection"
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db_temp",
        )
        # logging.info(f"Retrieving top {top_k} documents for query: {query}")
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    except Exception as e:
        logging.error(f"Error during document retrieval: {e}")
        raise


# Run ingestion only if needed
if __name__ == "__main__":
    directory_path = "./temp_data"
    try:
        vectorstore = ingest_files(directory_path)
        if vectorstore:
            logging.info("Document ingestion and embedding completed.")
        else:
            logging.info("No documents were ingested or embedded.")
    except Exception as e:
        logging.error(f"An error occurred during the ingestion process: {e}")

