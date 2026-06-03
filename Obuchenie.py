import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
# Настраиваем оптимизацию CUDA (чтобы не было проблем с памятью)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Тут начинается квантование, выбирается 4битное, с нормальным распределением, с 32 битной точностью и двойным квантованием для экономии памяти
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,     
    bnb_4bit_use_double_quant=True,
)
# Загружается токенизатор, берется модель из hugging face, разрешается нестандартный код нужный для qwen, паддинг справа
model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right"
)
# Если у токенизатора нет токена паддинга, то используем eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Список стоп токенов для остановки генерации, они не используются в обучении и вставлены для информации
stop_tokens = ["###", "User:", "Assistant:", "\n\n\n"]
tokenizer.eos_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token
# Загружается модель, применяется квантование, автоматическое распределение слоев на gpu или cpu
print("Загрузка модели")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",                        
    trust_remote_code=True,
    torch_dtype=torch.float32,              
)
# LoRA, нужен чтобы не обучать модель полностью, 32 ранг означает количество обучаемых параметров на слой,
# альфа, то есть коэффицент масштабирования, должен быть в 2 раз больше, 10 % нейронов отключаются для регуляризации
# обучаются лишь базовые слои: Запрос, Ключ, Значение, Вывод.
lora_config = LoraConfig(
    r=32,                  
    lora_alpha=64,           
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="lora_only",
)
# Применяем LoRA к модели, показываем, какая сколько парамметров имеет эта часть и каков ее масштаб
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Тут загружаются и форматируются данные, преобразуется json строка в словать, извлекаются вопрос и запрос, формируется простой промпт, 
def load_and_format_data(file_path):
    formatted_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            user_msg = data['messages'][0]['content']
            assistant_msg = data['messages'][1]['content']
            
            # Тот же формат промпта
            full_text = f"### User:\n{user_msg}\n\n### Assistant:\n{assistant_msg}\n###"
            
            formatted_data.append({"text": full_text})
    
    return formatted_data
# Превращаем текст в токены, обрезаем текст длиннее 512 токенов, паддинг до 512 токенов, максимальная длина последовательности тоже 512
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None,
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Загрузка и распределение данных, загружаются все данные, затем преобразуются в датасет, который затем токенизируется
all_data = load_and_format_data("training_data.jsonl")
dataset = Dataset.from_list(all_data)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
# Тут разделяется датасет на обучающую и валидационную, 80% обучающая, 20% валидационная
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Размер обучающей выборки: {len(train_dataset)}")
print(f"Размер валидационной выборки: {len(eval_dataset)}")
# Метрики качества, берется токен с максимальной вероятностью, считается количество совпадений
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    correct = (predictions == labels).sum()
    total = labels.size
    accuracy = correct / total
    
    return {
        "accuracy": accuracy,
        "perplexity": np.exp(1.0)  
    }
# Это отвечает за объединение отдельных примеров в батчи
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
# Параметры обучения
training_args = TrainingArguments(
    output_dir="./qwen2-sparql-lora",
    num_train_epochs=6, # количество эпох
    per_device_train_batch_size=1, # размер батча на 1 устройство
    gradient_accumulation_steps=8, # накопление градиентов, где эффективный батч 8
    learning_rate=1e-4, # скорость обучения
    warmup_steps=50,  # шаги разогрева для достижения заданной скорости
    warmup_ratio=0.15, # альтернативный способ задавания разогрева, не помешает
    logging_steps=10, # логгируются каждые 10 шагов
    save_steps=200, # сохраняяются чекпоинты каждые 200 шагов
    save_total_limit=3, # хранятся только 3 последних чекпоинта
    fp16=True, # половинная точность, формат для экономии памяти
    gradient_checkpointing=True, # экономия памяти
    remove_unused_columns=False, # не удаляь неисправные колонки
    weight_decay=0.05, # штраф за большие веса
    dataloader_pin_memory=False, # отключается фиксация памяти
    report_to="none", # не отправлять отчеты
    max_grad_norm=0.3, # обрезаются градиенты
    lr_scheduler_type="cosine", # тип планировщика скорости
)
# Создается трейнер и запускается обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,    
    eval_dataset=eval_dataset,    
    data_collator=data_collator,
    compute_metrics=compute_metrics, 
)
trainer.train()
# модель сохраняется
model.save_pretrained("./qwen-lora")
tokenizer.save_pretrained("./qwen-lora")
print("конец")
# Оценивается модель на валидационной выборке
eval_trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=1,
        report_to="none",
    ),
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

eval_results = eval_trainer.evaluate()
print(f"   Eval Loss: {eval_results['eval_loss']:.4f}")
