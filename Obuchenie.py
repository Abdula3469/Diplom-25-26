import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, #Загружает модель в 4битном формате
    bnb_4bit_quant_type="nf4", #формат 4битного квантования, точнее обычного
    bnb_4bit_compute_dtype=torch.float32,     
    bnb_4bit_use_double_quant=True, #Двойное квантование для дополнительной экономии места
)


#Загрузка токенизатора, который превращает текст в числа
model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

stop_tokens = ["###", "User:", "Assistant:", "\n\n\n"]
tokenizer.eos_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token

#Тут происходит загрузка модели,
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, #применяет квантование
    device_map="auto", #автоматически выбирает куда загружаться, на cpu или gpu                
    trust_remote_code=True,
    torch_dtype=torch.float32,              
)


lora_config = LoraConfig(
    r=64,   #Количество дополнительных параметров, чем этот параметр выше, тем точнее               
    lora_alpha=128,   #коэффицент масштабирования, желательно, чтобы было в 2 раз больше чем количество доп. параметров        
    lora_dropout=0.1, #Отключает часть, а точнее 10% нейронов для регуляризации
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #слои внимания, точнее Запрос, Ключ, Значение и Выход
    bias="lora_only",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#Тут происходит загрузка и форматирование данных
def load_and_format_data(file_path):
    formatted_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            user_msg = data['messages'][0]['content']
            assistant_msg = data['messages'][1]['content']
            
            full_text = f"### User:\n{user_msg}\n\n### Assistant:\n{assistant_msg}\n###"
            
            formatted_data.append({"text": full_text})
    
    return formatted_data
#Токенизация
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True, #Обрезает текст, который длиннее 512 токенов
        padding="max_length", #Добавляет паддинг до 512 токенов
        max_length=512, #Максимальная длина последовательности
        return_tensors=None,
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy() #Указывает какой токен правильный
    
    return tokenized

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
#Аргументы обучения
training_args = TrainingArguments(
    output_dir="./qwen2-sparql-lora",
    num_train_epochs=5, #Модель 5 раз пройдёт по всем данным, чтобы повысить качество
    per_device_train_batch_size=1, #размер батча, ввиду экономии очень маленький
    gradient_accumulation_steps=8, #накопление градиентов, эффективный батч
    learning_rate=2e-4, #стандартная скорость обучения
    warmup_steps=50, 
    warmup_ratio=0.1, #штрафует за большие веса
    logging_steps=10, 
    save_steps=200, 
    save_total_limit=3, 
    fp16=True, #Использует float16 для ускорения
    gradient_checkpointing=True, #Экономит память
    remove_unused_columns=False, 
    weight_decay=0.01, 
    dataloader_pin_memory=False, 
    report_to="none", 
    max_grad_norm=0.5, #Обрезает градиенты
)
#Запуск обучения
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
    
model.save_pretrained("./qwen-final")
tokenizer.save_pretrained("./qwen-final")
print("конец")
