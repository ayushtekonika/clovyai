import os
import logging
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory  # Import the memory module
from retrieve import retrieve_as_retriever  # Ensure this function loads the existing vectorstore
from dotenv import load_dotenv
#import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List

from langchain_core.chat_history import InMemoryChatMessageHistory

def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source: {os.path.basename(os.path.normpath(doc.metadata['source'])).replace("temp_data\\", "")}\nPage Number: {doc.metadata['page']}\nArticle Snippet: {doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


class ChatAssistant:    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.llm = ChatMistralAI(
                        model="mistral-large-latest",
                        temperature=0.2,
                        max_retries=2,
                    )


    def generate_response(self):

        retriever = retrieve_as_retriever()

        ### Contextualize question ###
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )


        ### Answer question ###
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain


    def Response(self, conversational_rag_chain, query, session_id):

        response = conversational_rag_chain.invoke(
                {"input": query},
                config={
                    "configurable": {"session_id": session_id}
                },  # constructs a key "abc123" in `store`.
            )
        
        response_with_references = f"{response["answer"]}\n`````{format_docs_with_id(response['context'])}"
        print("response", response)

        response = response["answer"]

        return response_with_references


def main():

    if 'assistant' not in st.session_state:
        st.session_state.assistant = ChatAssistant()
        st.session_state.conversational_rag_chain = st.session_state.assistant.generate_response()
    
    session_id = "123455"  # Static session ID for example purposes

    st.title("City of Lakewood AI-Powered Research Tool")
    st.write("(I have only ingested limited information on \"Food Safety\")")

    query = st.text_input("Your question:", placeholder="Type your question here...")
    if st.button("Submit") and query:
        with st.spinner("Generating response..."):
            response = st.session_state.assistant.Response(
                st.session_state.conversational_rag_chain, query, session_id
            )
            st.write(response)
    

if __name__=="__main__":
    main()
    

