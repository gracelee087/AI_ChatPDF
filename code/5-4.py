
# 전체구조의 2번째 :split구간 

#앞의 문서는 너무커서 문서/문장을/글자단위로 짤라버리자!! 나눠!! document transformer 문서에서 확인가능!

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # 가장 많이 씀 

#Loader
loader = PyPDFLoader("dbs.pdf")
pages = loader.load_and_split() # 문서 자를거얌 

#Split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,  # 몇글자 단위? 300글자!
    chunk_overlap  = 20, # 오버랩! 300글자 단위로 잘릴때 겹치게해서 단어가 짤려나가지 않도록 
    length_function = len, 
    is_separator_regex = False, # 문장 구분 기준 정규표현식으로 짜를껀지 
)
texts = text_splitter.split_documents(pages) # 자른문서를 또 잘라 

print(texts[5])   # 문장이 엄청 줄어듬. 