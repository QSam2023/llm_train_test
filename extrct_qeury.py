import jsonlines

query_list = []

with jsonlines.open("data/human_label_result_1208.jsonl") as reader:
    for d in reader:
        query = d["question"]
        print(query.replace("\n", " "))
