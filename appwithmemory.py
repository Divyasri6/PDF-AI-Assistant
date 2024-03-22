import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# pdf
from PyPDF2 import PdfReader
from fpdf import FPDF


from dotenv import load_dotenv
persist_directory='db1'

# Load environment variable
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# Combine text
def get_pdf_text(pdf_docs):
    text=""
    # iterate over all pdf files uploaded
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        # iterate over all pages in a pdf
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

# split texts into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # use the embedding object on the splitted text of pdf docs
    vector_index = Chroma.from_texts(text_chunks, embedding=embeddings,persist_directory=persist_directory)
    vector_index.persist()
    return vector_index


def get_conversational_chain(vectorstore):
    prompt_template = """
        You help everyone by answering questions as detailed as possible from the provided context, improve your answers from previous answers in History, and make sure to provide all the details.
        Don't try to make up an answer; if you don't know, just say that you don't know.
        If the answer is in a different language other than English, then translate it to English.
        PLEASE answer in English Language only.
        Do not say "Based on the information you provided, ..." or "I think the answer is...". Just answer the question directly in detail.

    Chat History:\n
    {chat_history}\n

    Context:\n 
    {context}?\n

    Question: \n
    {question}\n
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)

    # Define prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create retriever (assuming vectorstore is used as retriever)
    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":2})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # Create conversational chain with prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        combine_docs_chain_kwargs={"prompt": prompt},
        retriever=retriever,
        memory=memory,
        verbose=True,
    )
    return chain


def user_input(user_question):
    # Get the chat history from session state
    chat_history = st.session_state.get('chat_history', [])
    
    # Assuming you have a conversational chain object stored in session state
    chain = st.session_state.get('conversation')
    
    if chain:
        # Generate a response from the conversational chain
        response = chain({'question': user_question, 'chat_history': chat_history})
        
        # Append the response to the chat history
        chat_history.append({'role': 'user', 'content': user_question})
        chat_history.append({'role': 'AI', 'content': response["answer"]})
        
        # Update the chat history in session state
        st.session_state['chat_history'] = chat_history
        
        # Display the conversation
        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                st.write("User: ", message['content'])
            else:
                st.write("AI: ", message['content'])
        
    # Export chat history to PDF
        pdf = UnicodePDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        # Add both regular and bold variants of DejaVuSans font
        pdf.add_font('DejaVuSans', '', 'fonts\DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVuSans', 'B', 'fonts\DejaVuSans-Bold.ttf', uni=True)
        for message in st.session_state.chat_history :
            pdf.set_font("DejaVuSans", style='B')
            if message['role'] == 'user':
                role = "User"
            else:
                role = "AI"   
            pdf.cell(0, 10, role, ln=True)
            pdf.set_font("DejaVuSans")
            pdf.multi_cell(0, 10, message['content'], ln=True)
        pdf_output = pdf.output(dest='S')
        with open("chat_history.pdf", "wb") as f:
            f.write(pdf_output)
        st.download_button(
            label="Download Chat History as PDF", 
            data=bytes(pdf_output), 
            file_name="chat_history.pdf", 
            mime="application/pdf")
        
    else:
        st.error("Conversational chain not initialized. Please initialize the chain before using the chat.")

class UnicodePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Chat Messages', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

def main():
    st.set_page_config("Chat PDF")
   

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        
        user_input(user_question)
           
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #print(f"Type of raw_text: {type(raw_text)}")

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vector_store(text_chunks)

                conversation=get_conversational_chain(vectorstore)
                st.session_state.conversation=conversation
                st.session_state.memory = memory  # Save memory in session state
                st.session_state.vectorstore = vectorstore  # Save vectorstore in session state

                st.success("Done")
       
            
if __name__ == "__main__":
    main()