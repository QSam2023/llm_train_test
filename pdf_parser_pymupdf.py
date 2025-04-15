# coding: utf8

import fitz  # PyMuPDF

#import jieba
#from text_utils import jieba

from index_utils import *

def is_punc(char):
    return char in ",.!?，。？！"

def has_punc(text):
    return any([is_punc(c) for c in text])

def merged_paragraph(paragraph_list):
    if not paragraph_list:
        return []
    new_p_list = []
    new_p_list = [paragraph_list[0]]
    for i in range(len(paragraph_list))[1:]:
        last_p = new_p_list[-1]
        p = paragraph_list[i]
        new_p_list.append(p)
    return new_p_list

def analyze_pdf(file_path):
    doc = fitz.open(file_path)
    paragraph_list = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # 加载每一页
        blocks = page.get_text("dict")["blocks"]  # 获取文本块

        p = []
        for block in blocks:
            if "lines" in block:  # 检查文本块
                for line in block["lines"]:
                    for span in line["spans"]:  # 文本区域
                        text = span["text"]
                        if page_foot(text):
                            continue
                        p.append(text)
                paragraph = "".join(p).strip()
                if not page_foot(paragraph) and len(paragraph) > 0:
                    paragraph_list.append(paragraph)
                p = []
    doc.close()
    #return paragraph_list
    res = merged_paragraph(paragraph_list)
    #print(len(paragraph_list), len(res))
    return res

