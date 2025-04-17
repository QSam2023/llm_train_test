import os
import random
import json
import jsonlines

sample_num = 16000

file_dir = "data"
fn_list = [
    #"human_label_result_1208.jsonl",
    #"merged_eval_results_v2.jsonl",
    "answer.jsonl",
    "answer_w_recall.jsonl",
]

def get_system_prompt():
    prompt = f"""您是文档问答系统高级专家。
    根据以下要求执行任务，在文档中查找相关内容并回答问题；

    任务要求：
      1、输出一定在提供的文档内容中，且只能根据提供的文档内容来回答问题
      2、直接回答问题，不需要列出中间的思考过程，不需要拓展
      3、同一个问题，文本中有不同的答案，需要综合起来回答
      4、问题答案在文档中是分点/条、系列措施时列举时，需要全部列出不遗漏;
      5、文本中数值与两边的文字可能有空格或换行符，与没有这两类的符号的含义是相同的。例如 ""持有8926万股"" 与 ""持有 8926 万股"" 、""持有 8926万股""、""持有 8926 万股""含义是相同的
    """
    return prompt


def trans_chat_template(d):
    query = d.get("query", "")
    query = d.get("question", "") if not query else query

    ref = d.get("ref")
    ref = d.get("paragraphs", "") if not ref else ref
    ref = d.get("reference_content", "") if not ref else ref
    ref = d.get("result", "") if not ref else ref

    answer = d.get("answer", "")
    answer = d.get("ground_truth", "") if not answer else answer

    if not query or not ref or not answer:
        return []

    prompt = get_system_prompt()

    return [
        {
            "role": "user",
            "content": f"{prompt}\n\n文档内容：{ref}\n问题：{query}",
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]


data_list = []

for fn in fn_list:
    with jsonlines.open(os.path.join(file_dir, fn)) as reader:
        for line in reader:
            data_list.append(trans_chat_template(line))

print(len(data_list))

random.shuffle(data_list)

data_list = data_list[:sample_num]
output_data = [{"conversation": conv} for conv in data_list if conv]

with open("data/sft_16k_data.json", "w") as writer:
    jsonlines.Writer(writer).write_all(output_data)
    #json.dump(output_data, writer, ensure_ascii=False)

