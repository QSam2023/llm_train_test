import json
import random
import concurrent.futures
from tqdm import tqdm
from generrate_query_from_chunk import GenQuestion
from extract_chunk_file import load_chunk_file


def process_company(company_name, reference_list, output_file):
    gen_question = GenQuestion()
    question_ref_list = []
    for reference_content in reference_list:
        result = gen_question.llm_api_get_question(company_name, reference_content)
        question_ref = {
            "company_name": company_name,
            "reference_content": reference_content,
            "question_list": result,
        }
        question_ref_list.append(question_ref)
    return question_ref_list

def main():

    ref_chunk_dict = load_chunk_file(chunk_size=30, chunk_num=40)

    print("total company num: ", len(ref_chunk_dict))
    print("avg length of chunk_text: ", sum([sum(map(len, chunk)) / 40 for chunk in ref_chunk_dict.values()]) / len(ref_chunk_dict))


    ref_chunk_dict = load_chunk_file()
    output_file = "question_ref.jsonl"
    
    # 使用ThreadPoolExecutor进行并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ref_chunk_dict)) as executor:
        # 创建所有任务
        futures = []
        for company_name, reference_list in ref_chunk_dict.items():
            #reference_list = random.sample(reference_list, 2)
            future = executor.submit(process_company, company_name, reference_list, output_file)
            futures.append(future)
        
        # 使用tqdm显示进度
        all_results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"处理过程中出现错误: {e}")

        cost_token = 0
        # 将所有结果写入文件
        with open(output_file, "w", encoding="utf-8") as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                cost_token += len(result["reference_content"]) + "".join(result["question_list"]).count(" ")

        print(f"cost_token: {cost_token}")
        cost_money = cost_token / 1000 * 0.0003 * 0.7
        print(f"cost_money: {cost_money:.2f} RMB")

if __name__ == "__main__":
    main()
