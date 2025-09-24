from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
# 이전: from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 이전: from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# 이전: from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# 이전: from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("Choose a file")
st.write("---")

# 파일을 업로드하면 얘가 임시 디렉토리를 만들어서 tempdirectory파일에 업로드한 파일을 넣고, 여기서 저장된걸 가져와서 사용하고... loader로 불러오고, ... 아래 과정이 수행됨(tempfile, os는 이미 내장함수임
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
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
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma (영구 저장 하고싶을때!--> 이건 그냥 집에서 할때 쓰면됨. 우리가 하려는건 streamlit 상에서 임시 저장되서 거기서 계속 질문하고 하는거니깐, chroma를 쓰면안됨.)
    db = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma_db")
    # 이전: db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("Ask PDF")
    question = st.text_input('Please write your question')

    if st.button('Ask'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain.invoke({"query": question})
            # 이전: result = qa_chain({"query": question})
            st.write(result["result"])