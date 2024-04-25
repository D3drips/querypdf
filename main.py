import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
os.environ['GOOGLE_API_KEY'] =""
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
def get_pdf_text(pdfdoc):
    text= ""
    for pdf in pdfdoc:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text        

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks =text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore= FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore    

def main():
    st.set_page_config(page_title="Project work based on Query PDF",page_icon=":books:")
    st.header("Project work based on Query PDF")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
            user_input(user_question)
    
    with st.sidebar:
        st.subheader("Upload the PDF")
        pdfdoc=st.file_uploader("upload your PDF here and Click on the process",accept_multiple_files=True)
        

        
        if st.button("Process"):
           with st.spinner("Processing"):
               
               raw_text= get_pdf_text(pdfdoc)
               
               text_chunks= get_text_chunks(raw_text)
               
               vectorstore= get_vectorstore(text_chunks)
               st.success("Done")
               st.write(vectorstore)
                
if __name__ == "__main__":
    main()
    
    







