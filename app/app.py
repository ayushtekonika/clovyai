from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

if groq_key is None:
    print(groq_key, "groq")
    os.environ["GROQ_API_KEY"] = groq_key

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

def get_summary(query: str) -> str:

    messages = [
    (
        "system",
        "The human query that is provided is a conversation between a medical expert and a patient. so You are a medical expert AI agent summarizer equipped with comprehensive knowledge. Your primary function is to summarize the patient's query by providing accurate summary. Your responses are concise, reliable, and presented in an easy-to-understand format. Make sure the response is in key points",
    ),
    ("human", query)
    ]
    
    response = llm.invoke(messages)
    if hasattr(response, 'content') and response.content:
        return response.content
    else:
        return ""