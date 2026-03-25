import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./phi2-sparql-lora-final")

print("СЛИЯНИЕ ГУРРЕН-ЛАГАНН")
merged_model = model.merge_and_unload()

output_dir = "./phi2-sparql-merged"
merged_model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f"Модель объединеня {output_dir}")