import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
import warnings
warnings.filterwarnings("ignore")

class SparqlAssistant:
    def __init__(self, adapter_path="./phi2-sparql-lora-final3"):
        print("Вроде работает")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.stop_token_ids = [
            self.tokenizer.encode("###", add_special_tokens=False)[0],
            self.tokenizer.encode("User:", add_special_tokens=False)[0],
            self.tokenizer.eos_token_id,
        ]
        
        print("Модель Загружена")
    
    def extract_first_sparql(self, text: str) -> str:
        """извлекает первый корректный SPARQL запрос"""
        
        pattern = r'(PREFIX\s+wd:\s*<[^>]+>\s+PREFIX\s+wdt:\s*<[^>]+>.*?SELECT\s+\?answer\s+WHERE\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            sparql = match.group(1).strip()
            if sparql.count('{') == sparql.count('}'):
                return sparql

        if 'SELECT' in text and 'WHERE' in text:
            start = text.find('PREFIX')
            if start == -1:
                start = text.find('SELECT')

            brace_count = 0
            end = start
            in_where = False
            
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                    in_where = True
                elif char == '}':
                    brace_count -= 1
                    if in_where and brace_count == 0:
                        end = i + 1
                        break
            
            return text[start:end].strip()
        
        return None
    
    def generate(self, question: str) -> str:
        prompt = f"### User:\n{question}\n\n### Assistant:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                forced_eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Assistant:\n" in response:
            response = response.split("### Assistant:\n")[-1]
        
        if "###" in response:
            response = response.split("###")[0]
        
        sparql = self.extract_first_sparql(response)
        
        if sparql is None:
            lines = response.split('\n')
            result = []
            for line in lines:
                result.append(line)
                if line.strip() == '}' and len(result) > 5:
                    break
            sparql = '\n'.join(result)
        
        return sparql.strip()

if __name__ == "__main__":
    assistant = SparqlAssistant()
    print("Введите exit для выхода")
    while True:
        question = input("Вопрос: ").strip()
        
        if question.lower() == 'exit':
            print("Выход")
            break
        
        if question:
            try:
                result = assistant.generate(question)
                print(f"\nsparql запрос:\n{result}\n")
            except Exception as e:
                print(f"Ошибка: {e}")
