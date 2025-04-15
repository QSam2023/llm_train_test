# coding: utf8

import re
import hashlib


NUM_DICT = {c:str(i+1) for i, c in enumerate(list("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"))}

def num2normal(text):
    new_char_list = []
    for char in text:
        if char in NUM_DICT:
            new_char_list.append(NUM_DICT[char])
            new_char_list.append("、")
        else:
            new_char_list.append(char)
    return "".join(new_char_list)

def fullwidth_to_halfwidth(text):
    new_text = ""
    for char in text:
        code = ord(char)
        # 全角字符（除空格外）的编码在区间 0xFF01-0xFF5E
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        # 全角空格特殊处理（全角空格为 0x3000）
        elif code == 0x3000:
            code = 0x0020
        new_text += chr(code)
    return new_text

def remove_whitespace(text):
    return "".join([char for char in text if char !=" "])

def gen_uid(text):
    return hashlib.md5(text.encode("utf")).hexdigest()

def is_catalogue(text):
    text = remove_whitespace(text)
    return re.search(r"[\.]{7}[0-9]+", text)

def page_foot(text):
    text = text.strip()
    if re.match(r"^[0-9]{1,3}-[0-9]+-[0-9]+$", text):
        return True
    if re.match(r"^[0-9]{1,3}—[0-9]+—[0-9]+$", text):
        return True
    #word_list = text.strip().split()
    if re.match(r"^[0-9]+-[0-9]+-[0-9]+-[0-9]+$", text):
        return True
    if text[-5:] == "招股意向书":
        return True
    if text[-5:] == "招股说明书":
        return True
    return False

def simple_clean(text):
    text = text.strip()
    text = re.sub(r"：", ":", text)
    text = re.sub(" ", "", text)
    return text

def extract_org_name(last_text, text):
    final_res = None

    word_list = text.strip().split()

    # simple_clean
    text = simple_clean(text)
    last_text = simple_clean(last_text)

    # from page foot
    if len(word_list) == 2 and word_list[-1][-5:] == "招股意向书":
        final_res = word_list[0]

    try:
        if re.search(r"发行人简要情况|发行人概况", last_text):
            final_res = text.split(":")[-1]

        if re.search(r"发行人:.*有限公司$", text):
            final_res = re.sub(r".*发行人:", "", text).strip()

        if re.search(r"公司名称:|中文名称:|中文:|发行人:", text[:5]):
            final_res = text.split(":")[-1]

        if re.search(r"中文名称.*有限公司$", text):
            final_res = text

        if re.search(r"发行.*公司名称:*有限公司", text):
            final_res = text.split(":")[-1].replace(r"有限公司.*", "有限公司")

        if re.search(r"英文名称|英文:", text):
            final_res = last_text.split(":")[-1]

    except:
        pass

    if final_res is not None:
        final_res = re.sub(r"招股意向书|.*名称", "", final_res)
        final_res = re.sub(r"公司.*", "公司", final_res)
        if len(final_res) < 4 or "有限公司" not in final_res or "指" in final_res:
            final_res = ""
        final_res = final_res.strip(":")

    return final_res if final_res else None


"""
第五节 发行人基本情况
三、公司历史沿革及股本形成情况

(一)公司的历史沿革
1、东莞勤上五金塑胶制品有限公司设立
"""


def is_title_1(text):
    text = text.strip()
    if re.match(r"^第[一二三四五六七八九十]+节", text):
        return True
    if re.match(r"^第[一二三四五六七八九十]+章", text):
        return True
    return False

def is_title_2(text):
    text = text.strip()
    if re.match(r"^[一二三四五六七八九十]+、", text):
        return True
    return False

def is_title_3(text):
    text = text.strip()
    if re.match(r"^\([一二三四五六七八九十]+\)", text):
        return True
    return False

def check_valid_title(text):
    text = text.strip()
    if len(text.split()) > 1:
        return False
    if text[-1] in ":；。：%0123456789;":
        return False
    return True

def is_title_4(text):
    text = text.strip()
    if re.match(r"^[0-9]+、", text):
        return True
    return False


def spam_content(content):
    count_digit = sum([1 for char in content if char in "0123456789%"])
    if len(content) <= 1:
        return True
    if len(content) > 10 and len(set(list(content))) <= 3:
        return True
    if count_digit >= 20:
        return True
    return False

def low_content(content):
    count_digit = sum([1 for char in content if char in "0123456789%"])
    if  count_digit >= 5 and count_digit / len(content) > 0.8:
        return True
    if len(content) <= 2:
        return True
    return False

def find_pattern(text):
    pattern = r'\d\s[\u4e00-\u9fff]'
    matches = re.finditer(pattern, text)
    rule1 = [(match.start(), match.end()) for match in matches]

    pattern = r'[年月]\s\d'
    matches = re.finditer(pattern, text)
    rule2 = [(match.start(), match.end()) for match in matches]

    pattern = r'[\u4e00-\u9fff]\s\d'
    matches = re.finditer(pattern, text)
    rule3 = [(match.start(), match.end()) for match in matches]

    return rule1 + rule2 + rule3

def new_find_pattern(text):
    pattern = r'\d\s[年月万日号元个亿名项原人一台中天家第股类吨室]'
    matches = re.finditer(pattern, text)
    rule1 = [(match.start(), match.end()) for match in matches]

    pattern = r'[年月为司计至度于和的第目额金产款本到入资在日路润自费较及备用号长有币加过率人股行台了机比注达益值约元品件由类程]\s\d'
    matches = re.finditer(pattern, text)
    rule2 = [(match.start(), match.end()) for match in matches]
    return rule1 + rule2


def remove_digit_whitespace(text):
    match_positions = find_pattern(text)

    l = list(text)
    remove_list = []

    for s, e in match_positions:
        for i in range(s, e):
            if l[i] == " ":
                remove_list.append(i)

    return "".join([c for i, c in enumerate(l) if i not in remove_list])

