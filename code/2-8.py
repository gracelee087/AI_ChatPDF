from dotenv import load_dotenv
load_dotenv()

# LLM 또는 CHAT MODEL이 있음!! 

from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI()
result = chat_model.predict("hi")
print(result)