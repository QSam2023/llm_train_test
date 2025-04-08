import json
import jsonlines
import tqdm
from utils import get_system_prompt

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

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
    for d in tqdm(reader):
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

        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")


        output_ids = model.generate(
            input_ids,
            max_new_tokens = 1024,
            do_sample = False,
            top_p = 0.9,
            top_k = 0
        )

        answer = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        res = {
            "question": query,
            "answer": answer,
        }
        output_f.write(json.dumps(res, ensure_ascii=False) + "\n")

output_f.close()
