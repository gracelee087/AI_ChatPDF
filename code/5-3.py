from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("dbs.pdf")
pages = loader.load_and_split()

print(pages[5])