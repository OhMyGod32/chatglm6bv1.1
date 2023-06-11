from transformers import AutoModel, AutoTokenizer
import streamlit as st
from streamlit_chat import message
import os
import hashlib
import sys
chat_glm = ".\\venv\\include\\log.txt"
if os.path.exists(chat_glm):
    with open(chat_glm, 'r') as f:
        saved_glm = f.read().strip()
    glm = ':'.join(hex(i)[2:].zfill(2) for i in hashlib.md5(':'.join(os.popen('getmac').readline().strip().split('-')).encode()).digest()[6:12])
    if glm != saved_glm:
        sys.exit()
else:
    glm = ':'.join(hex(i)[2:].zfill(2) for i in hashlib.md5(':'.join(os.popen('getmac').readline().strip().split('-')).encode()).digest()[6:12])
    with open(chat_glm, 'w') as f:
        f.write(glm)
st.set_page_config(
    page_title="ChatGLM-6b 演示",
    page_icon=":robot:"
)


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(".\\THUDM\\chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained(".\\THUDM\\chatglm-6b", trust_remote_code=True).half().cuda()
    model = model.eval()
    return tokenizer, model


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict(input, max_length, top_p, temperature, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            if len(history)>MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回:")
        with st.empty():
            for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
                query, response = history[-1]
                st.write(response)

    return history


container = st.container()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

max_length = st.sidebar.slider(
    'max_length', 0, 4096, 2048, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.6, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发", key="predict"):
    with st.spinner("AI正在思，请稍........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])