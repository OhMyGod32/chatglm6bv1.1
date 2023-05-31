from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
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
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer

import os
import hashlib
import sys      
chat_glm = '.\venv\include\log.txt'
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

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(".\\THUDM\\chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained(".\\THUDM\\chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
