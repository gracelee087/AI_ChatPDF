__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
# 이전: from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 이전: from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# 이전: from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# 이전: from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

button(username="5RIWuyzpfX", floating=True, width=221)

#Title
st.title("ChatPDF")
st.write("---")

#Get OpenAI API Key input
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

#File upload
uploaded_file = st.file_uploader("Please upload a PDF file!",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#Code that runs when file is uploaded
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into Chroma (persistent storage)
    db = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma_db")
    # Previous: db = Chroma.from_documents(texts, embeddings_model)

    #Create Handler to receive Stream
    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)

    #Question
    st.header("Ask questions to PDF!!")
    question = st.text_input('Please enter your question')

    if st.button('Ask Question'):
        with st.spinner('Wait for it...'):
            chat_box = st.empty()
            stream_hander = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            qa_chain.invoke({"query": question})
            # Previous: qa_chain({"query": question})