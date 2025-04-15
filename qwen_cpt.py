import os

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments


os.environ["WANDB_PROJECT"] = "my-awesome-project"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


max_seq_length = 8196 # Supports RoPE Scaling interally, so choose any!
base_moodel = "Qwen/Qwen2.5-1.5B"
# Get LAION dataset
url = "data/corpus_data.json"
dataset = load_dataset("json", data_files=url, split="train")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_moodel,
    max_seq_length = max_seq_length,
    full_finetuning = True,
)

# 对齐Qwen的格式内容
#tokenizer = get_chat_template(
#    tokenizer,
#    chat_template = "chatml",
#    map_eos_token = True,
#)

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    return {"text" : [example + EOS_TOKEN for example in examples["text"]],}



trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        warmup_ratio = 0.1,
        num_train_epochs = 1,

        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer.train()

model.save_pretrained_merged("outputs/model_cpt", tokenizer)
