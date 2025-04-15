import os

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

os.environ["WANDB_PROJECT"] = "my-awesome-project"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


max_seq_length = 8196 # Supports RoPE Scaling interally, so choose any!
# Get LAION dataset
url = "./data/demo_data.json"
dataset = load_dataset("json", data_files=url, split="train")



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    full_finetuning = True,
)

# 对齐Qwen的格式内容
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    map_eos_token = True,
)

def formatting_prompts_func(examples):
    convos = examples["conversation"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }


dataset = dataset.map(formatting_prompts_func, batched = True,)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        max_steps = 100,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
        report_to="wandb",
        logging_steps=10,
        save_steps=100,
        run_name="qwen_lora_training"
    ),
)
trainer.train()

model.save_pretrained_merged("outputs/vllm_model_sft", tokenizer, save_method = "merged_16bit",)

