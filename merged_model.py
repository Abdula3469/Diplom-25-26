import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
#Тут загружается базовая модель, загружается на cpu
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)
#Тут загружается лора адаптер, он содержит только изменения
model = PeftModel.from_pretrained(base_model, "./qwen-lora")
#Тут же складываются веса базовой модели и лора адаптера, чтобы создать модель без лора слоев
merged_model = model.merge_and_unload()
#Тут же сохраняется новая модель, которая готова для конвертации в gguf, чтоьы  использовать ее в ollama
output_dir = "./qwen-merged"
merged_model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f"Модель объединеня {output_dir}")
