import jsonlines

fn = "gen_data/answer.jsonl"

with jsonlines.open(fn) as reader:
    for line in reader:
        print(line["question"])
