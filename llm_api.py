import os
import numpy as np
from typing import List

from openai import OpenAI


volc_client = OpenAI(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


def volcengine_llm_api(text):
    completion = volc_client.chat.completions.create(
		#model="doubao-1-5-pro-32k-250115",
		model="doubao-1-5-lite-32k-250115",
		messages=[
			{"role": "user", "content": text},
		],
    )

    return completion.choices[0].message.content

def volcengine_llm_api_pro(text):
    completion = volc_client.chat.completions.create(
		model="doubao-1-5-pro-32k-250115",
		messages=[
			{"role": "user", "content": text},
		],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    import time
    start_time = time.time()
    text = "1+1=?"
    print(volcengine_llm_api(text))
    end_time = time.time()
    print(f"time: {end_time - start_time}s")

