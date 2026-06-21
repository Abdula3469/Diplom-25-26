import tkinter as tk
from tkinter import scrolledtext, messagebox
import subprocess
import threading
import datetime
import os
import requests
import re

class WikidataRAG:
    def __init__(self):
        self.wikidata_api = "https://www.wikidata.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SPARQL-Assistant/1.0',
            'Accept': 'application/json'
        })
        self.cache = {}
    
    def search_entities(self, search: str, limit: int = 3):
        cache_key = f"search_{search}_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            params = {
                'action': 'wbsearchentities',
                'search': search,
                'language': 'ru',
                'format': 'json',
                'limit': limit,
                'type': 'item'
            }
            
            response = self.session.get(self.wikidata_api, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('search', []):
                    results.append({
                        'id': item['id'],
                        'label': item.get('label', search),
                        'description': item.get('description', ''),
                    })
                
                self.cache[cache_key] = results
                return results
            return []
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []
    
    def extract_entities_from_question(self, question: str) -> dict:
        
        question_lower = question.lower()
        
        question_patterns = {
    'country': {
        'keywords': ['страна', 'государство', 'какая страна', 'в какой стране'],
        'main_type': 'entity',
        'secondary_type': 'country',
        'property': 'P17'
    },
    
    'capital': {
        'keywords': ['столица', 'главный город'],
        'main_type': 'country',
        'secondary_type': 'city',
        'property': 'P36'
    },
    
    'author': {
        'keywords': ['кто написал', 'автор', 'создал', 'написал'],
        'main_type': 'work',
        'secondary_type': 'author',
        'property': 'P50'
    },
    
    'administrative_unit': {
        'keywords': ['административная единица', 'регион', 'область', 'край', 'республика'],
        'main_type': 'entity',
        'secondary_type': 'administrative_unit',
        'property': 'P131' 
    },
    
    'birth_place': {
        'keywords': ['где родился', 'место рождения', 'родился'],
        'main_type': 'person',
        'secondary_type': 'place',
        'property': 'P19'
    },
    
    'creator': {
        'keywords': ['создатель', 'кто создал', 'основал'],
        'main_type': 'entity',
        'secondary_type': 'creator',
        'property': 'P170' 
    },
    
    'location': {
        'keywords': ['место нахождения', 'где находится', 'расположен', 'находится'],
        'main_type': 'entity',
        'secondary_type': 'location',
        'property': 'P276' 
    },
    
    'composer': {
        'keywords': ['композитор', 'кто написал музыку', 'написал музыку'],
        'main_type': 'musical_work',
        'secondary_type': 'composer',
        'property': 'P86'
    },
    
    'performer': {
        'keywords': ['исполнитель', 'кто исполнил', 'поет', 'певец', 'певица'],
        'main_type': 'musical_work',
        'secondary_type': 'performer',
        'property': 'P175'
    },
    
    'director': {
        'keywords': ['режиссёр', 'кто снял', 'постановщик'],
        'main_type': 'film',
        'secondary_type': 'director',
        'property': 'P57'
    },
    
    'instance_of': {
        'keywords': ['что такое', 'кто такой', 'является', 'это'],
        'main_type': 'entity',
        'secondary_type': 'instance',
        'property': 'P31'
    },
    
    'founder': {
        'keywords': ['основатель', 'кто основал'],
        'main_type': 'organization',
        'secondary_type': 'founder',
        'property': 'P112'
    },
    
    'currency': {
        'keywords': ['денежная единица', 'валюта'],
        'main_type': 'country',
        'secondary_type': 'currency',
        'property': 'P38'
    },
    
    'citizenship': {
        'keywords': ['гражданство', 'подданство', 'гражданин'],
        'main_type': 'person',
        'secondary_type': 'citizenship',
        'property': 'P27'
    },
    
    'spouse': {
        'keywords': ['супруг', 'супруга', 'муж', 'жена', 'кто был женат'],
        'main_type': 'person',
        'secondary_type': 'spouse',
        'property': 'P26'
    },
    
    'death_date': {
        'keywords': ['дата смерти', 'когда умер', 'год смерти'],
        'main_type': 'person',
        'secondary_type': 'date',
        'property': 'P570'
    },
    
    'birth_date': {
        'keywords': ['дата рождения', 'когда родился', 'год рождения'],
        'main_type': 'person',
        'secondary_type': 'date',
        'property': 'P569'
    },
    
    'killer': {
        'keywords': ['кто убил', 'убийца'],
        'main_type': 'victim',
        'secondary_type': 'killer',
        'property': 'P157'
    },
    
    'occupation': {
        'keywords': ['профессия', 'кем работает', 'кем был', 'род деятельности'],
        'main_type': 'person',
        'secondary_type': 'occupation',
        'property': 'P106'
    },
    
    'height': {
        'keywords': ['высота', 'рост', 'какой высоты', 'сколько метров'],
        'main_type': 'entity',
        'secondary_type': 'height',
        'property': 'P2044'
    },
    
    'population': {
        'keywords': ['население', 'жителей', 'сколько людей', 'численность'],
        'main_type': 'entity',
        'secondary_type': 'population',
        'property': 'P1082'
    }
}
        
        question_type = None
        for qtype, pattern in question_patterns.items():
            if any(kw in question_lower for kw in pattern['keywords']):
                question_type = qtype
                break
        
        stop_words = {'кто', 'что', 'где', 'когда', 'почему', 'как', 'какой', 
                     'какая', 'какое', 'какие', 'сколько', 'куда', 'откуда',
                     'зачем', 'чей', 'чья', 'чье'}
        
        clean_question = question_lower
        for word in stop_words:
            clean_question = clean_question.replace(word, '')
        
        for qtype, pattern in question_patterns.items():
            for kw in pattern['keywords']:
                clean_question = clean_question.replace(kw, '')
        
        clean_question = re.sub(r'[^\w\s]', ' ', clean_question)
        clean_question = re.sub(r'\s+', ' ', clean_question).strip()
        
        words = clean_question.split()
        
        result = {
            'main_entity': None,
            'secondary_entity': None,
            'property_hint': None,
            'question_type': question_type,
            'found_count': 0
        }
        
        if question_type:
            pattern = question_patterns[question_type]
            result['property_hint'] = pattern['property']
            
            for n in range(min(3, len(words)), 0, -1):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    if len(phrase) > 2:
                        entities = self.search_entities(phrase, limit=1)
                        if entities:
                            result['main_entity'] = {
                                'qid': entities[0]['id'],
                                'label': entities[0]['label'],
                                'search': phrase
                            }
                            break
                if result['main_entity']:
                    break
            
            if not result['main_entity']:
                entities = self.search_entities(question, limit=1)
                if entities:
                    result['main_entity'] = {
                        'qid': entities[0]['id'],
                        'label': entities[0]['label'],
                        'search': question
                    }
        
        else:
            for n in range(min(3, len(words)), 0, -1):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    if len(phrase) > 2:
                        entities = self.search_entities(phrase, limit=1)
                        if entities:
                            result['main_entity'] = {
                                'qid': entities[0]['id'],
                                'label': entities[0]['label'],
                                'search': phrase
                            }
                            break
                if result['main_entity']:
                    break
        
        if result['main_entity']:
            result['found_count'] = 1
        
        return result
    
    def get_search_context(self, question: str) -> dict:
        entities = self.extract_entities_from_question(question)
        
        return {
            'found': entities['found_count'] > 0,
            'main_entity': entities['main_entity'],
            'property_hint': entities['property_hint'],
            'question_type': entities['question_type'],
            'search_query': entities['main_entity']['search'] if entities['main_entity'] else question
        }


class SparqlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen-sparql")
        self.root.geometry("950x800")
        self.root.resizable(True, True)
        
        self.model_name = "qwen-2026"
        self.rag = WikidataRAG()
        
        self.log_file = "sparql_assistant.log"
        self._init_log_file()
        
        self.setup_ui()
    
    def setup_ui(self):
        colors = {
            'bg': '#f5f5f5',
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'info': '#e8f4f8',
            'warning': '#fef9e7'
        }
        
        self.root.configure(bg=colors['bg'])
        
        title_frame = tk.Frame(self.root, bg=colors['primary'], height=50)
        title_frame.pack(fill="x")
        
        title = tk.Label(title_frame, text="SPARQL Assistant", 
                        font=("Arial", 16, "bold"), fg="white", bg=colors['primary'])
        title.pack(pady=12)
        
        status_frame = tk.Frame(self.root, bg=colors['bg'])
        status_frame.pack(fill="x", padx=15, pady=(10, 5))
        
        self.status_model = tk.Label(status_frame, text=f"Модель: {self.model_name}", 
                                     font=("Arial", 9), bg=colors['bg'], fg=colors['primary'])
        self.status_model.pack(side=tk.LEFT)
        
        main_frame = tk.Frame(self.root, bg=colors['bg'])
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        tk.Label(main_frame, text="Вопрос:", font=("Arial", 11, "bold"), 
                bg=colors['bg']).pack(anchor="w", pady=(0, 5))
        
        self.question_text = tk.Text(main_frame, height=4, font=("Arial", 11), 
                                     wrap="word", relief="solid", borderwidth=1)
        self.question_text.pack(fill="x", pady=(0, 10))
        
        btn_frame = tk.Frame(main_frame, bg=colors['bg'])
        btn_frame.pack(fill="x", pady=(0, 10))
        
        self.generate_btn = tk.Button(btn_frame, text="Модель генерирует SPARQL", 
                                      font=("Arial", 12, "bold"), command=self.generate,
                                      bg=colors['secondary'], fg="white", height=2)
        self.generate_btn.pack(side=tk.LEFT, fill="x", expand=True)
        
        self.clear_btn = tk.Button(btn_frame, text="Очистить", 
                                   font=("Arial", 11), command=self.clear,
                                   bg=colors['primary'], fg="white", height=2)
        self.clear_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        self.status_label = tk.Label(main_frame, text="", font=("Arial", 9), 
                                     bg=colors['bg'], fg=colors['primary'])
        self.status_label.pack()
        
        context_frame = tk.LabelFrame(main_frame, text="RAG контекст (найдено в Wikidata)", 
                                      font=("Arial", 9, "bold"), bg=colors['bg'])
        context_frame.pack(fill="x", pady=(10, 5))
        
        self.context_text = tk.Text(context_frame, height=5, font=("Arial", 9), 
                                    wrap="word", bg=colors['info'], relief="flat")
        self.context_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        prompt_frame = tk.LabelFrame(main_frame, text="Промпт к модели", 
                                     font=("Arial", 9, "bold"), bg=colors['bg'])
        prompt_frame.pack(fill="x", pady=(5, 5))
        
        self.prompt_text = tk.Text(prompt_frame, height=4, font=("Arial", 9), 
                                   wrap="word", bg=colors['warning'], relief="flat")
        self.prompt_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        result_frame = tk.LabelFrame(main_frame, text="SPARQL запрос", 
                                     font=("Arial", 10, "bold"), bg=colors['bg'])
        result_frame.pack(fill="both", expand=True, pady=(5, 0))
        
        result_btn_frame = tk.Frame(result_frame, bg=colors['bg'])
        result_btn_frame.pack(fill="x", padx=5, pady=5)
        
        self.copy_btn = tk.Button(result_btn_frame, text="Копировать", 
                                  font=("Arial", 9), command=self.copy_sparql,
                                  bg=colors['success'], fg="white")
        self.copy_btn.pack(side=tk.LEFT, padx=2)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=10, 
                                                     font=("Courier", 11), wrap="word")
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"ЛОГ ОБУЧЕНИЯ МОДЕЛИ SPARQL\n")
                f.write(f"Создан: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _log(self, level, message, sparql=None, error=None, prompt=None):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{level}]\n")
            f.write(f"  Сообщение: {message}\n")
            if prompt:
                f.write(f"  Промпт:\n{prompt}\n")
            if sparql:
                f.write(f"  SPARQL ответ:\n{sparql}\n")
            if error:
                f.write(f"  Ошибка: {error}\n")
            f.write("-" * 50 + "\n")
    
    def copy_sparql(self):
        text = self.result_text.get("1.0", tk.END).strip()
        if text and not text.startswith("Ошибка"):
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="Скопировано", fg="green")
            self.root.after(2000, lambda: self.status_label.config(text="", fg="gray"))
    
    def clear(self):
        self.question_text.delete("1.0", tk.END)
        self.context_text.delete("1.0", tk.END)
        self.prompt_text.delete("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)
        self.status_label.config(text="")
    
    def generate(self):
        question = self.question_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Внимание", "Пожалуйста, введите вопрос")
            return
        
        self._log("INFO", f"Пользовательский запрос: {question}")
        
        self.generate_btn.config(state="disabled")
        self.copy_btn.config(state="disabled")
        self.status_label.config(text="RAG: поиск в Wikidata...", fg="orange")
        self.result_text.delete("1.0", tk.END)
        self.context_text.delete("1.0", tk.END)
        self.prompt_text.delete("1.0", tk.END)
        
        thread = threading.Thread(target=self._process, args=(question,))
        thread.daemon = True
        thread.start()
    
    def _process(self, question):
        try:
            context = self.rag.get_search_context(question)
            
            if context['found']:
                context_info = f"""Найдено в Wikidata:
• Тип вопроса: {context['question_type'] or 'общий'}
• Основная сущность: {context['main_entity']['label']} (QID: {context['main_entity']['qid']})
• Найдено по запросу: '{context['main_entity']['search']}'
• Рекомендуемое свойство: {context['property_hint'] or 'модель определит сама'}"""
                
                self._log("INFO", f"RAG нашел: {context['main_entity']['label']} ({context['main_entity']['qid']})")
                
                if context['property_hint']:
                    prompt = f"""Ты эксперт по SPARQL запросам для Wikidata.

Вопрос: {question}

Найдена сущность в Wikidata:
- Название: {context['main_entity']['label']}
- QID: {context['main_entity']['qid']}

Рекомендуемое свойство: {context['property_hint']}

Сгенерируй SPARQL запрос:
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?answer WHERE {{ wd:{context['main_entity']['qid']} wdt:{context['property_hint']} ?answer . }}

Только SPARQL запрос, без пояснений."""
                else:
                    prompt = f"""Ты эксперт по SPARQL запросам для Wikidata.

Вопрос: {question}

Найдена сущность в Wikidata:
- Название: {context['main_entity']['label']}
- QID: {context['main_entity']['qid']}

Сгенерируй SPARQL запрос, используя QID {context['main_entity']['qid']}.
Ты сам определи нужное свойство.

Формат: PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT ?answer WHERE {{ wd:{context['main_entity']['qid']} wdt:P??? ?answer . }}

Только SPARQL запрос."""
            else:
                context_info = f"""Сущность не найдена в Wikidata
• Поисковый запрос: '{context['search_query']}'"""
                
                prompt = f"""Ты эксперт по SPARQL запросам для Wikidata.

Вопрос: {question}

Сущность не найдена, используй шаблон:
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?answer WHERE {{ wd:Q??? wdt:P??? ?answer . }}

Только SPARQL запрос."""
            
            self.root.after(0, self._update_context, context_info)
            self.root.after(0, self._update_prompt, prompt)
            self.root.after(0, self.status_label.config, {"text": "Модель генерирует...", "fg": "orange"})
            
            result = subprocess.run(
                ['ollama', 'run', self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                clean_output = self._clean_sparql(output)
                
                self._log("SUCCESS", f"Модель сгенерировала SPARQL", sparql=clean_output, prompt=prompt)
                self.root.after(0, self._update_result, clean_output)
                self.root.after(0, self.status_label.config, {"text": "Готово", "fg": "green"})
            else:
                error_msg = result.stderr.strip()
                self._log("ERROR", f"Ошибка модели", error=error_msg)
                self.root.after(0, self._update_result, f"Ошибка: {error_msg}")
                self.root.after(0, self.status_label.config, {"text": "Ошибка", "fg": "red"})
                
        except Exception as e:
            self.root.after(0, self._update_result, f"Ошибка: {e}")
            self.root.after(0, self.status_label.config, {"text": "Ошибка", "fg": "red"})
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state="normal"))
            self.root.after(0, lambda: self.copy_btn.config(state="normal"))
    
    def _clean_sparql(self, text: str) -> str:
        text = re.sub(r'wdQ(\d+)', r'wd:Q\1', text)
        text = re.sub(r'wdtP(\d+)', r'wdt:P\1', text)
        
        patterns = [
            r'(PREFIX\s+wd:.*?PREFIX\s+wdt:.*?SELECT\s+\?answer\s+WHERE\s*\{[^}]+\})',
            r'(SELECT\s+\?answer\s+WHERE\s*\{[^}]+\})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                if 'PREFIX' not in result:
                    result = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
""" + result
                return result
        
        return text
    
    def _update_context(self, text):
        self.context_text.delete("1.0", tk.END)
        self.context_text.insert("1.0", text)
    
    def _update_prompt(self, text):
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", text)
    
    def _update_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)

if __name__ == "__main__":
    root = tk.Tk()
    app = SparqlGUI(root)
    root.mainloop()
