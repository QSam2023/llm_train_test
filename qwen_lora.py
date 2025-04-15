import os

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

os.environ["WANDB_PROJECT"] = "my-awesome-project"
os.environ["WANDB_LOG_MODEL"] = ""
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


max_seq_length = 8196 # Supports RoPE Scaling interally, so choose any!
# Get LAION dataset
url = "./data/sft_16k_data.json"
#url = "./data/demo_data.json"
dataset = load_dataset("json", data_files=url, split="train")



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
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

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 2,
        warmup_steps = 50,
        max_steps = 500,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
        report_to="wandb",
        logging_steps=50,
        save_steps=250,
        run_name="qwen_lora_training"
    ),
)
trainer.train()

model.save_pretrained_merged("outputs/vllm_model_3b_16k", tokenizer, save_method = "merged_16bit",)
