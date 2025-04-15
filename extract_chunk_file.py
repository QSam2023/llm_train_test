import os
import jsonlines 
import numpy as np

def merged_chunk_text(chunk_group):
    return "\n".join([chunk["content"] for chunk in chunk_group])


def load_chunk_file(path="pdf_chunk_txt", chunk_size=30, chunk_num=40):
    path = "pdf_chunk_txt"

    ref_dict = {}
    ref_chunk_dict = {}

    for fn in os.listdir(path):
        company_name = fn.split(".")[0]
        ref_dict[company_name] = []
        with jsonlines.open(os.path.join(path, fn), "r") as f:
            for line in f:
                ref_dict[company_name].append(line)
        start = int(len(ref_dict[company_name]) * 0.02)
        end = int(len(ref_dict[company_name]) * 0.9)
        ref_dict[company_name] = ref_dict[company_name][start:end]

    for company_name, chunks in ref_dict.items():
        ref_chunk_dict[company_name] = []
        # 计算可能的连续chunk组合数量
        total_possible_chunks = len(chunks) - chunk_size + 1
        
        # 如果可用的chunk组合数量小于需要的数量，则使用所有可能的组合
        if total_possible_chunks <= chunk_num:
            selected_indices = range(total_possible_chunks)
        else:
            # 均匀采样chunk_num个位置
            selected_indices = np.linspace(0, total_possible_chunks - 1, chunk_num, dtype=int)
        
        # 对每个选中的位置，取连续的chunk_size个chunk并拼接
        selected_chunks = []
        for idx in selected_indices:
            chunk_group = chunks[idx:idx + chunk_size]
            # 将连续的chunk拼接成一个字符串
            chunk_group_text = merged_chunk_text(chunk_group)
            selected_chunks.append(chunk_group_text)
        
        ref_chunk_dict[company_name] = selected_chunks
    
    return ref_chunk_dict



if __name__ == "__main__":
    ref_chunk_dict = load_chunk_file()
    for company_name, chunks in ref_chunk_dict.items():
        print(company_name)
        for chunk_sample in chunks[:10]:
            print(chunk_sample)
            print("-"*100)
        break