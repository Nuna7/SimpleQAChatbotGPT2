# Question-Answering Chatbot

This is a simple **Question-Answering** chatbot built using **LangChain**, **Hugging Face models**, and **Streamlit**. The system allows users to upload a PDF document and ask questions based on its content. The chatbot can either provide answers using the content of the document or answer general questions if no document is uploaded.

## How it Works:
- **With PDF Upload**: If a PDF is uploaded, the system will retrieve relevant parts of the document to provide answers.
- **Without PDF**: The system uses a pre-trained language model to generate answers based on the question alone.

## Installation

To install the necessary dependencies, simply run the following command:

```bash
pip install streamlit langchain transformers langchain-community sentence-transformers faiss-gpu faiss-cpu torch
```
## Usage
1. Run the Streamlit app:

```bash
streamlit run main.py
```

2. In the browser:
- Upload a PDF document (optional).
- Enter your question and the chatbot will generate a response based on the PDF (if uploaded) or from the model alone.

## How the Prompt Helps the Model
The prompt template provides a structured format to the **LLM (Language Model)** to ensure the response is precise, based only on the provided context, or returns a fallback message when insufficient information is available.

1. The template guides the LLM to answer concisely and avoid unnecessary information.
2. It instructs the model to refrain from repeating the question and explicitly mentions when context is insufficient to answer.
   
By using this specific prompt structure, the model stays aligned with the goal of generating useful, focused responses based on the provided context, or clearly stating when it can't answer due to lack of context.
