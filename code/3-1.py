from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI()

content = "k-pop"

result = chat_model.predict(content + "에 대한 시를 써줘")
print(result)