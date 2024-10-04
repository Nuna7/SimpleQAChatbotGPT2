"""
pip install streamlit langchain transformers langchain-community sentence-transformers faiss-gpu faiss-cpu torch
"""

import streamlit as st
from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import tempfile
import torch

def load_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name
        
    loader = PyPDFLoader(temp_pdf_path) 
    documents = loader.load()  
    return documents

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=512,  
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    return HuggingFacePipeline(pipeline=generator)

def create_retriever(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)  
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) 
    return retriever

def truncate_text(text, max_tokens):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens)
    return text


def get_prompt_template():
    template = """
    You are an AI assistant tasked with answering questions based on the provided context. 
    Please follow these guidelines:
    1. Your answer should be based strictly on the information given in the context.
    2. If the context doesn't contain relevant information to answer the question, state: "I'm sorry, but I don't have enough information in the provided context to answer this question."
    3. Be concise and precise in your response.
    4. Do not repeat the question or include any part of this instruction in your answer.

    Context: {context}

    Question: {question}

    Answer: """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    return prompt

def post_process_answer(answer):
    if "Answer:" in answer:
        answer = answer.split("Answer:")[1].strip()
    
    lines = answer.split('\n')
    processed_lines = [line for line in lines if not line.startswith(('Context:', 'Question:', 'Answer:', 'You are', 'Please follow'))]
    
    processed_answer = ' '.join(processed_lines).strip()
    
    # If the answer is empty
    if not processed_answer:
        return "I'm sorry, but I couldn't generate a relevant answer based on the provided context."
    
    return processed_answer

def main():
    st.title("Question-Answering System with PDF Context")
    
    question = st.text_input("Enter your question:")
    uploaded_pdf = st.file_uploader("Upload a PDF (optional)", type="pdf")  
    
    documents = None
    if uploaded_pdf is not None:
        with st.spinner("Loading and processing PDF..."):
            documents = load_pdf(uploaded_pdf)

    if question:
        model = load_model()
        
        if documents:
            retriever = create_retriever(documents)
            truncated_question = truncate_text(question, max_tokens=512)
            context_docs = retriever.get_relevant_documents(truncated_question)
            context = truncate_text(" ".join([doc.page_content for doc in context_docs]), max_tokens=512)
            
            if not context:
                context = "There is no additional context available to answer this question."
            
            prompt_template = get_prompt_template()
            llm_chain = LLMChain(llm=model, prompt=prompt_template)
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain, 
                document_variable_name="context"
            )
            qa_chain = RetrievalQA(
                retriever=retriever,  
                combine_documents_chain=stuff_chain
            )

            try:
                answer = qa_chain.invoke({"query": question, "context": context})  
                processed_answer = post_process_answer(answer['result'])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                processed_answer = "I'm sorry, but I encountered an error while processing your question. Please try again with a simpler query or shorter context."
        
        else:
            try:
                prompt = f"Question: {question}\nAnswer: "
                response = model(prompt)  
                processed_answer = post_process_answer(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                processed_answer = "I'm sorry, but I encountered an error while generating your answer. Please try again with a different query."

        st.write("### Answer:")
        st.write(processed_answer)

            
if __name__ == "__main__":
    main()

        