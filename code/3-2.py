from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI()

st.title('한국 ai 산업의 미래')

content = st.text_input('산업의 미래 먹거리를 제시해주세요.')

if st.button('보고서 작성 요청하기'):
    with st.spinner('보고서 작성 중...'):
        result = chat_model.predict(content + "에 대한 보고서를 써줘")
        st.write(result)