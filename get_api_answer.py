import json
import jsonlines
import concurrent.futures
from tqdm import tqdm
from llm_api import volcengine_llm_api
from llm_api import volcengine_llm_api_pro

from utils import get_system_prompt



#input_file = "question_ref.jsonl"
#output_file = "answer.jsonl"

input_file = "rag_data/recall_result.jsonl"
output_file = "rag_data/answer_w_recall.jsonl"


batch_request = []

"""
with jsonlines.open(input_file, "r") as reader:
    for line in reader:
        for question in line["question_list"]:
            batch_request.append({
                "question": question,
                "reference_content": line["reference_content"],
                "company_name": line["company_name"],
            })
"""

with jsonlines.open(input_file, "r") as reader:
    for line in reader:
        batch_request.append({
            "question": line["query"],
            "reference_content": line["result"],
        })

def process_request(request):
    prompt = get_system_prompt()
    message = f"{prompt}\n\n文档内容：{request['reference_content']}\n问题：{request['question']}"

    response = volcengine_llm_api_pro(message)
    result = {
        "query": request["question"],
        "result": request["reference_content"],
        "answer": response,
    }

    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for request in batch_request:
        future = executor.submit(process_request, request)
        futures.append(future)

    all_results = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            all_results.append(result)
        except Exception as e:
            print(f"处理过程中出现错误: {e}")

    with jsonlines.open(output_file, "w") as writer:
        for result in all_results:
            writer.write(result)

