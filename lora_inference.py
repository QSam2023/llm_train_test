import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

model_name = "outputs/checkpoint-60"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    dtype = torch.bfloat16,
)

FastLanguageModel.for_inference(model)

message = [{"role": "user", "content": "您是文档问答系统高级专家。\n    根据以下要求执行任务，在文档中查找相关内容并回答问题；\n\n    任务要求：\n      1、输出一定在提供的文档内容中，且只能根据提供的文档内容来回答问题\n      2、直接回答问题，不需要列出中间的思考过程，不需要拓展\n      3、同一个问题，文本中有不同的答案，需要综合起来回答\n      4、问题答案在文档中是分点/条、系列措施时列举时，需要全部列出不遗漏;\n      5、文本中数值与两边的文字可能有空格或换行符，与没有这两类的符号的含义是相同的。例如 \"\"持有8926万股\"\" 与 \"\"持有 8926 万股\"\" 、\"\"持有 8926万股\"\"、\"\"持有 8926 万股\"\"含义是相同的\n    \n\n文档内容：\n问题：深圳麦格米特 发行人产品主要应用哪些领域"}]


text_streamer = TextStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True)

input_ids = tokenizer.apply_chat_template(
    message,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")


_ = model.generate(
    input_ids,
    streamer = text_streamer,
    max_new_tokens = 1024,
    do_sample = False,
    top_p = 0.9,
    top_k = 0
)
