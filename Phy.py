#Тут мы обуччаем модель
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Загрузка модели")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def load_and_format_data(file_path):
    """Загружает данные и форматирует для обучения"""
    formatted_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            
            user_msg = data['messages'][0]['content']
            assistant_msg = data['messages'][1]['content']
            
            
            
            full_text = f"### User:\n{user_msg}\n\n### Assistant:\n{assistant_msg}"
            
            formatted_data.append({"text": full_text})
    
    return formatted_data

def tokenize_function(examples):
    """Токенизирует текст и создает labels для обучения"""
    
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None,
    )
    
    
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Чекпоинт проверки кода")
train_data = load_and_format_data("training_data.jsonl")
dataset = Dataset.from_list(train_data)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./phi2-sparql-lora",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print(f"Размер: {len(tokenized_dataset)} примеров")
print(f"Шаги: {len(tokenized_dataset) * 5} (5 эпох)\n")

try:
    trainer.train()
    
    print("\nСохраняемся")
    model.save_pretrained("./phi2-sparql-lora-final")
    tokenizer.save_pretrained("./phi2-sparql-lora-final")
    
except Exception as e:
    print(f"\nОшибка: {e}")