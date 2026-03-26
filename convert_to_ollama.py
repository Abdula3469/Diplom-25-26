import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
#Тут загружается базовая модель, загружается на кпу
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)
#Тут загружается лора адаптер, он содержит только изменения
model = PeftModel.from_pretrained(base_model, "./phi2-sparql-lora-final")
#Тут же я складываю веса базовой модели и лора адаптера, чтобы создать модель без лора слоев
merged_model = model.merge_and_unload()
#Тут же сохраняется новая модель, которая готова для конвертации в gguf, чтоьы  испоьзовать ее в оллама
output_dir = "./phi2-sparql-merged"
merged_model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f"Модель объединеня {output_dir}")
