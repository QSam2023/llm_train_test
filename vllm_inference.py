import json
import jsonlines
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import get_system_prompt

# vLLM加载方式：可以是本地路径或 huggingface repo
# 注意：需要配合你要使用的模型权重
model_name_or_path = "outputs/vllm_model"
llm = LLM(model=model_name_or_path, tensor_parallel_size=1)

eval_fn = "data/merged_eval_results_v2.jsonl"
output_fn = "eval_result.jsonl"

# 打开输出文件
output_f = open(output_fn, "w", encoding="utf-8")

with jsonlines.open(eval_fn) as reader:
    data = list(reader)

batch_size = 32
num_data = len(data)

# vLLM的生成参数，可自行调节
sampling_params = SamplingParams(
    max_tokens=1024,     # 最多生成多少个token
    temperature=0.0,     # 可根据需求调整
    top_p=0.9,
    top_k=-1,
    # repetition_penalty=1.0, 等等
)

# 这里是一个示例函数，用于生成可读的prompt文本
# 你可以根据自己项目的模板来拼接
def build_prompt(ref, query):
    system_text = get_system_prompt()
    prompt = (
        f"{system_text}\n\n"  # system层
        f"文档内容：{ref}\n"   # context
        f"问题：{query}"      # user问题
    )
    return prompt

# 分批处理
for i in tqdm(range(0, num_data, batch_size)):
    batch_data = data[i:i+batch_size]

    # 构造一批 prompts
    prompts = []
    for d in batch_data:
        query = d.get("query", "")
        ref = d.get("ref", "")
        # ground_truth = d.get("answer", "") # 看需求是否要用
        prompt_text = build_prompt(ref, query)
        prompts.append(prompt_text)

    # 调用 vLLM 一次性生成所有结果
    # outputs 是一个列表，对应每个 prompt
    outputs = llm.generate(prompts, sampling_params)

    # 写出结果
    for d, output in zip(batch_data, outputs):
        # 每个output.generations 里可能包含多个采样解；默认是1个
        answer_text = output.outputs[0].text
        result = {
            "question": d.get("query", ""),
            "answer": answer_text.strip()
        }
        output_f.write(json.dumps(result, ensure_ascii=False) + "\n")

output_f.close()
