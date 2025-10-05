import os
from openai import OpenAI
import json

client = OpenAI(
    api_key="sk-f0a5c4226f37482d852970bf3c47589d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def reason(model, prompt_ , temperature):

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model=model,
        temperature = temperature,        
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_},
        ],
        timeout=6000,
        extra_body={"enable_thinking": False}
        
    )
    return json.loads(completion.model_dump_json())["choices"][0]["message"]["content"]

