# coding: utf8

import os
import json
import hashlib
from collections import Counter
import time
from typing import List, Dict, Any

from pdf_parser_pymupdf import analyze_pdf
from index_utils import *

#from milvus_utils import doc_info_insert


def gen_paragrah_docs(fn):
    doc_list = []

    org_name = []
    part_level_1 = "开头部分"
    part_level_2 = ""
    part_level_3 = ""
    part_level_4 = ""

    line_no = 0
    last_text = ""
    is_prefix = True

    org_name = None
    org_name_cand = None

    for line in analyze_pdf(fn):

        text = fullwidth_to_halfwidth(line)
        text = num2normal(text)


        org_name_cand = extract_org_name(last_text, text)


        if org_name_cand is not None and org_name is None:
            org_name = org_name_cand

        last_text = text

        line_no += 1

        if is_title_1(text):
            part_level_1 = text
            part_level_2 = ""
            part_level_3 = ""
            part_level_4 = ""

            #if "释义" in text:
            #    is_prefix = False

        if is_title_2(text) and check_valid_title(text):
            part_level_2 = text
            part_level_3 = ""

        if is_title_3(text) and check_valid_title(text):
            part_level_3 = text
            part_level_4 = ""

        if is_title_4(text) and check_valid_title(text):
            part_level_4 = text

        doc_info = {
            "id": "{}_{}".format(gen_uid(fn), line_no),
            "doc_id": "{}".format(fn.split("/")[-1]),
            #"line_no": line_no,
            "content": text,
            #"title_0": org_name,
            "title_1": part_level_1,
            "title_2": part_level_2,
            "title_3": part_level_3,
            "title_4": part_level_4,
        }
        doc_list.append(doc_info)

    final_org_name = org_name
    #c = Counter(org_name)
    #if org_name:
    #    final_org_name = sorted([(-v, len(k), k) for k, v in c.items()])[0][-1]

    line_no = 0
    new_doc_list = []
    for doc_info in doc_list:
        text = doc_info["content"]

        if text == final_org_name or is_catalogue(text) or page_foot(text):
            continue

        if len(text) <= 2 and new_doc_list:
            new_doc_list[-1]["content"] += text
            continue

        line_no += 1
        doc_info["title_0"] = final_org_name
        doc_info["line_no"] = line_no
        doc_info["doc_id"] = final_org_name


        new_doc_list.append(doc_info)

    #print(final_org_name)

    return new_doc_list


def chinese_hash_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# 修改主程序部分
if __name__ == "__main__":
    import os
    import time
    from argparse import ArgumentParser

    parser = ArgumentParser(description="parser pdf")

    parser.add_argument("--index_name", type=str)

    args = parser.parse_args()
    index_name = args.index_name

    directory = "raw_pdf"

    fn_list = [os.path.join(directory, filename) \
                   for filename in os.listdir(directory) \
                   if os.path.isfile(os.path.join(directory, filename))
                   and filename.lower().endswith(".pdf")
              ]

    #out_f = open(out_fn, "w")
    random_id = 1

    #print(time.asctime())
    print("Start to parser pdf....")
    #print("Output file is {}".format(out_fn))

    #log_f = open("/Users/fryzhang/WorkSpace/rag_project/hybrid_search_rag/insert_v1.log", "w")

    # 原循环代码中收集数据
    for fn in fn_list:
        try:
            doc_list = gen_paragrah_docs(fn)
            for doc_info in doc_list:
                doc_info["random_id"] = random_id
                random_id += 1
                #out_f.write("{}\n".format(json.dumps(doc_info, ensure_ascii=False)))

            start_time = time.time()
            doc_name = doc_list[0]["title_0"]
            print(doc_name)
            with open("pdf_chunk_txt/{}.jsonl".format(doc_name), "w") as f:
                for doc_info in doc_list:
                    f.write("{}\n".format(json.dumps(doc_info, ensure_ascii=False)))

        except:
            print("Error, {}".format(fn))
