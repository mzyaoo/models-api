import uuid
from datetime import datetime
from urllib.request import Request

from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pydantic import BaseModel
from typing import List
import time
# 启动 FastAPI 服务
import uvicorn

from qwen.model.QwenMessage import QwenMessage

# 初始化 FastAPI 应用
app = FastAPI()

# 在启动时加载模型和 tokenizer
model_dir = "/Library/MyFolder/Models/Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    starttime = time.time()
    response = await call_next(request)
    endtime = time.time()
    totaltime = endtime - starttime
    logmessage = f"request time : {request_time}, request method : {request.method}, request url : {request.url.path}, request status code : {response.status_code}, expend time : {totaltime:.2f}s seconds"
    print(logmessage)
    return response


class QwenRequest(BaseModel):
    stream: bool = False
    top_p: float = 0.9
    temperature: float = 0.7
    max_new_tokens: int = 256
    model: str = "qwen-chat"
    messages: List[QwenMessage]


@app.post("/qwen/chat")
async def chat(messages: List[QwenMessage]):
    prompt = messages[-1].content
    print("用户输入：", prompt)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 定义生成器函数，用于流式返回
    def generate_stream():
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True
            )

            # 获取新生成的 token（排除输入部分）
            generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]

            for token in generated_tokens:
                word = tokenizer.decode([token.item()], skip_special_tokens=True)
                if word.strip():
                    yield word

    # 返回流式响应
    return StreamingResponse(generate_stream(), media_type="text/plain")


# 响应头（每一段）
def format_openai_stream_response(token: str, index: int = 0, role=None, finish_reason=None):
    chunk = {
        "id": f"chatcmpl-{str(uuid.uuid4())[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen-chat",
        "choices": [
            {
                "index": index,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ]
    }
    if role:
        chunk["choices"][0]["delta"]["role"] = role
    if token:
        chunk["choices"][0]["delta"]["content"] = token
    return f"data: {chunk}\n\n"


@app.post("/v1/qwen/chat")
async def chatV1(request: QwenRequest):
    prompt = request.messages[-1].content
    print("用户输入：", prompt)

    text = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    def stream_generator():
        yield format_openai_stream_response("", role="assistant")  # 开头角色

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                output_scores=True,
                return_dict_in_generate=True
            )

            generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]

            for token in generated_tokens:
                word = tokenizer.decode([token.item()], skip_special_tokens=True)
                if word.strip():
                    yield format_openai_stream_response(word)

        yield format_openai_stream_response("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # 非流式返回完整内容
        full_text = ""
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                output_scores=True,
                return_dict_in_generate=True
            )

            generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]

            for token in generated_tokens:
                word = tokenizer.decode([token.item()], skip_special_tokens=True)
                full_text += word

        return {
            "id": f"chatcmpl-{str(uuid.uuid4())[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text
                    },
                    "finish_reason": "stop"
                }
            ]
        }


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
