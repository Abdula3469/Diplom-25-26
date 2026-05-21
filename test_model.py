import json
import subprocess
import re
import time
import os
from datetime import datetime

class SparqlOllamaTester:
    def __init__(self, model_name="qwen-sparql999.gguf"):
        self.model_name = model_name
        print(f"Используется модель: {self.model_name}")
        self._check_model()

    # Проверяет, что модель вообще существует в Ollama
    def _check_model(self):
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if self.model_name in result.stdout:
                print(f"Модель найдена")
                return True
            else:
                print(f"Модель не найдена")
                return False
        except Exception as e:
            print(f"Ошибка: {e}")
            return False
        
    # Извлекаем SPARQL запрос из ответа модели
    def extract_sparql(self, text: str) -> str:
        if not text:
            return None
        
        pattern = r'(PREFIX\s+wd:\s*<[^>]+>\s+PREFIX\s+wdt:\s*<[^>]+>.*?SELECT\s+\?answer\s+WHERE\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        if 'SELECT' in text and 'WHERE' in text:
            start = text.find('PREFIX')
            if start == -1:
                start = text.find('SELECT')
            if start == -1:
                return None
            
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
            
            if end > start:
                return text[start:end].strip()
        
        return None
    # тут проверяется, имеет ли запрос валидную sparql-структуру
    def has_valid_syntax(self, sparql: str) -> bool:
        if not sparql:
            return False
        
        # Проверяем наличие ключевых элементов структуры без учета префиксов
        has_select = 'SELECT' in sparql and '?answer' in sparql
        has_where = 'WHERE' in sparql and '{' in sparql and '}' in sparql
        
        return has_select and has_where
    
    # Нормализует запрос для сравнения, т.е. убираем префиксы, пробелы
    def normalize_for_comparison(self, sparql: str) -> str:
        if not sparql:
            return ""
        
        sparql = re.sub(r'PREFIX\s+wd:\s*<[^>]+>\s*', '', sparql)
        sparql = re.sub(r'PREFIX\s+wdt:\s*<[^>]+>\s*', '', sparql)
        sparql = re.sub(r'\s+', ' ', sparql)
        sparql = sparql.strip()
        
        return sparql
    # Это же отвечает за анализ результата генерации
    def analyze_result(self, generated: str, expected: str) -> dict:
        if generated is None:
            return {"type": "error", "message": "Не получилось извлечь sparql"}
        
        if not self.has_valid_syntax(generated):
            return {"type": "syntax_error", "message": "Тут синтаксическая ошибка"}
        
        # Сравниваем содержание без префиксов, ведь они на точность запроса не влияют
        gen_norm = self.normalize_for_comparison(generated)
        exp_norm = self.normalize_for_comparison(expected)
        
        if gen_norm == exp_norm:
            return {"type": "match", "message": "Совпадает"}
        else:
            return {"type": "mismatch", "message": "Синтаксис верен, запрос отличается"}
        
    # Отправляет запрос к Ollama
    def query_ollama(self, question: str, timeout: int = 60) -> dict:
        try:
            start_time = time.time()
            result = subprocess.run(
                ['ollama', 'run', self.model_name, question],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                return {"success": True, "response": result.stdout.strip(), "time": elapsed}
            else:
                return {"success": False, "error": result.stderr.strip(), "time": elapsed}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout", "time": timeout}
        except FileNotFoundError:
            return {"success": False, "error": "Ollama не найден", "time": 0}
        except Exception as e:
            return {"success": False, "error": str(e), "time": 0}
        
    # Тестируется модель на тестовом датасет
    def test_dataset(self, test_file_path: str, limit: int = None):
        print(f"\nЗагрузка: {test_file_path}")
        
        if not os.path.exists(test_file_path):
            print(f"Файл не найден")
            return
        
        # Тут происходит загрузка данных из датасета
        test_data = []
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data['messages'][0]['content']
                    expected = data['messages'][1]['content']
                    test_data.append((question, expected))
                except json.JSONDecodeError as e:
                    print(f"Ошибка в строке {line_num}: {e}")
        
        if not test_data:
            print("Нет данных, проверьте содержимое датасета")
            return
        
        if limit:
            test_data = test_data[:limit]
        
        print(f"Примеров в тестовой выборке: {len(test_data)}")
        
        results = []
        total_time = 0
        stats = {"match": 0, "mismatch": 0, "syntax_error": 0, "error": 0}
        
        for i, (question, expected) in enumerate(test_data, 1):
            print(f"\n[{i}/{len(test_data)}] {question[:55]}...")
            
            result = self.query_ollama(question)
            
            if not result["success"]:
                stats["error"] += 1
                results.append({
                    "question": question,
                    "expected": expected,
                    "generated": None,
                    "match": False,
                    "reason": f"Ошибка: {result['error']}",
                    "time": result["time"]
                })
                print(f"Ошибка: {result['error'][:60]}")
                continue
            
            generated = self.extract_sparql(result["response"])
            analysis = self.analyze_result(generated, expected)
            
            is_match = (analysis["type"] == "match")
            
            results.append({
                "question": question,
                "expected": expected,
                "generated": generated,
                "match": is_match,
                "reason": analysis["message"],
                "time": result["time"]
            })
            
            stats[analysis["type"]] = stats.get(analysis["type"], 0) + 1
            total_time += result["time"]
            
            # Тут выводится статус
            if analysis["type"] == "match":
                print(f" {analysis['message']} ({result['time']:.2f}с)")
            elif analysis["type"] == "mismatch":
                print(f" {analysis['message']} ({result['time']:.2f}с)")
            elif analysis["type"] == "syntax_error":
                print(f" {analysis['message']} ({result['time']:.2f}с)")
            else:
                print(f" {analysis['message']} ({result['time']:.2f}с)")
        
        # Здесь подводятся итоги
        print("Результаты тестирования отображаются тут:")
        print(f"Запрос составлен идеально:        {stats.get('match', 0)}/{len(test_data)} ({stats.get('match', 0)/len(test_data)*100:.1f}%)")
        print(f"Синтаксис запроса верен:  {stats.get('mismatch', 0)}")
        print(f"Синтаксис запроса неверен: {stats.get('syntax_error', 0)}")
        print(f"Ошибки запроса:   {stats.get('error', 0)}")
        print(f"Среднее время выполнения запроса:    {total_time/len(test_data):.2f}с")
        
        # Происходит сохранение отчета
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nОтчет: {report_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Просого запуска недостаточно, введите в консоль: python test_model.py test_data.jsonl")
        sys.exit(1)
    
    tester = SparqlOllamaTester("qwen-sparql999.gguf")
    tester.test_dataset(sys.argv[1])maTester("qwen-sparql999.gguf")
    tester.test_dataset(sys.argv[1])
