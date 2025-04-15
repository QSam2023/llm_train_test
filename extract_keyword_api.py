import json
import jsonlines
from tqdm import tqdm
from llm_api import volcengine_llm_api
from concurrent.futures import ThreadPoolExecutor, as_completed

fn = "gen_data/answer.jsonl"

question_ans = []
with jsonlines.open(fn) as reader:
    for line in reader:
        question = line["question"]
        answer = line["answer"]
        question_ans.append({"question": question, "answer": answer})

prompt_template = """
你是一名金融行业算法工程师
需要你开发一套金融AI检索系统，根据query的关键词，检索相关内容，从而获取问题答案。

给定你对应的query，请提取相关的关键词
同时针对answer，同样提取问答内容的关键信息，来衡量query检索的召回率

要求：
1. 公司名称不算做关键词，判定为停用词，忽略
2. 语气等疑问助词忽略，数字内容保持完整性

返回格式,请采用json格式
```json
{{
    "query_keywords": ["关键词1", "关键词2", "关键词3"],
    "answer_keywords": ["关键词1", "关键词2", "关键词3"]
}}
```

给定的query
{query}

给定的answer
{answer}

"""

def process_single_item(item):
    question = item["question"]
    answer = item["answer"]
    
    prompt = prompt_template.format(query=question, answer=answer)
    response = volcengine_llm_api(prompt)
    
    try:
        json_str = response.split("```json")[1].split("```")[0].strip()
        result = json.loads(json_str)
        item.update(result)
        return item
    except Exception as e:
        print(f"解析json出错: {e}")
        return None

keywords_res = []

# 使用线程池并发处理请求
with ThreadPoolExecutor(max_workers=20) as executor:
    # 提交所有任务
    future_to_item = {executor.submit(process_single_item, item): item for item in question_ans}
    
    # 使用tqdm显示进度
    for future in tqdm(as_completed(future_to_item), total=len(question_ans)):
        result = future.result()
        if result:
            keywords_res.append(result)

with open("gen_data/keywords_res.jsonl", "w") as writer:
    for res in keywords_res:
        writer.write(json.dumps(res, ensure_ascii=False) + "\n")
