# 본격적으로 질문하기 / pdf기반으로 체인 연결했기 때문에.. 이제 문맥에 맞게 대답을 받음. 

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

#Loader
loader = PyPDFLoader("dbs.pdf")
pages = loader.load_and_split()

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

# load it into Chroma (영구 저장)
db = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma_db")
# 이전: db = Chroma.from_documents(texts, embeddings_model)

#Question
question = "what is the main AI strategy of DBS?"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever()) # 체인을 통해서 결과받기
result = qa_chain.invoke({"query": question})
# 이전: result = qa_chain({"query": question})
print(result)