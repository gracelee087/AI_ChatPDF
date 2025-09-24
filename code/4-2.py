import streamlit as st
from langchain.llms import CTransformers  # llama-2-7b 개발자가 쓴 언어가 c++이여서 바꿔주는 거 임포트 해야함 

# 이제 OpenAI() 를 가져올 이유가 없음! 왜? 로컬에서 쓰니깐!

llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama"
)

st.title('organisational transformation')  # 주제

content = st.text_input('시의 주제를 제시해주세요.')  # 프롬트엔지니어링 

if st.button('시 작성 요청하기'):
    with st.spinner('시 작성 중...'):
        result = llm.predict("write a poem about " + content + ": ") # 이건 이제 llm 
        st.write(result)