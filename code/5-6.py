# 전체구조의 4,5,번째(물어보고 ...)

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

# load it into Chroma
db = Chroma.from_documents(texts, embeddings_model)

#Question
question = "what is the main AI strategy of DBS?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)
# 이전:
# retriver_from_llm = MultiQueryRetriever.from_llm(
#     retriever=db.as_retriever(), llm=llm
# )

docs = retriever_from_llm.get_relevant_documents(query=question)
print(len(docs)) # 어떤 문서가져왔는지 보기 
print(docs) # 결과 보기 
# 이전: print(result) # 결과 보기 