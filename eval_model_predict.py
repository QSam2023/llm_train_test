from unsloth import FastLanguageModel
import json
import jsonlines
from tqdm import tqdm
from utils import get_system_prompt

import torch
from transformers import TextStreamer

model_name = "outputs/checkpoint-60"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    dtype = torch.bfloat16,
)

FastLanguageModel.for_inference(model)

eval_fn = "data/merged_eval_results_v2.jsonl"
output_fn = "eval_result.jsonl"


output_f = open(output_fn, "w")

with jsonlines.open(eval_fn) as reader:
    data = list(reader)

batch_size = 32
num_data = len(data)

for i in tqdm(range(0, num_data, batch_size)):
    batch_data = data[i:i+batch_size]

    messages = []
    for d in batch_data:
        query = d.get("query", "")
        ref = d.get("ref", "")
        ground_truth = d.get("answer", "")

        prompt = get_system_prompt()

        message = [
            {
                "role": "user",
                "content": f"{prompt}\n\n文档内容：{ref}\n问题：{query}",
            }
        ]

    batch_input = tokenizer.apply_chat_template(
        message,
        add_generation_prompt = True,
        return_tensors = "pt",
        padding = True,
        truncation = True,
    ).to("cuda")

        # 添加 attention_mask
    attention_mask = (batch_input != tokenizer.pad_token_id).long()

    output_ids = model.generate(
        batch_input,
        max_new_tokens = 1024,
        attention_mask=attention_mask,  # 关键修复点
        do_sample = False,
        top_p = 0.9,
        top_k = 0
    )

    input_lengths = []
    for idx in range(batch_size.shape[0]):
        valid_token_idx = (batch_input[idx] != tokenizer.pad_token_id).nonzero()
        input_lengths.append(valid_token_idx.shape[0])

    for idx, d in enumerate(batch):
        offset = input_lengths[idx]
        answer_ids = output_ids[idx][offset:]

        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        res = {
            "question": d.get("query", ""),
            "answer": answer,
        }
        output_f.write(json.dumps(res, ensure_ascii=False) + "\n")

output_f.close()
