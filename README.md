# Генерация SPARQL-запросов с использованием дообученной LLM
Интеллектуальный сервис для автоматической генерации SPARQL-запросов к графу знаний Wikidata по вопросам на русском языке. Сервис работает на основе дообученной модели Qwen2-1.5B с использованием LoRA и 4-битного квантования, что позволяет запускать его на обычном персональном компьютере.

# Назначение
Сервис предназначен для автоматического преобразования вопросов на русском языке в синтаксически корректные и исполнимые SPARQL-запросы к базе знаний Wikidata. Позволяет пользователям, не знакомым с языком запросов SPARQL, получать структурированные данные из графа знаний.

# Инструкция запуска

Этот проект посвящен объединению и запуску модели для генерации SPARQL-запросов с использованием Ollama.

## 1. Подготовка
1. Скачайте и разархивируйте папку с проектом.
2. Откройте разархивированную папку в **Visual Studio Code (VSC)** или любом другом текстовом редакторе.

## 2. Установка зависимостей
Откройте терминал в в текстовом редакторе и выполните следующую команду для установки необходимых библиотек:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft
pip install bitsandbytes scipy safetensors
pip install requests jsonlines pandas
pip install numpy scikit-learn matplotlib
pip install sentence-transformers chromadb  # для RAG
```

## 3. Обучение и подготовка модели
Запустите последовательно следующие скрипты:
1. **Obuchenie.py** (обучение модели):
   ```bash
   python Obuchenie.py
   ```
2. **merged_model.py** (Объединение Базовой модели и обученных слоев):
   ```bash
   python merged_model.py
   ```

## 4. Конвертация в формат GGUF
Для работы с Ollama модель нужно конвертировать с помощью `llama.cpp`.

1. Склонируйте репозиторий `llama.cpp`:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   ```
2. Запустите конвертацию:
   ```bash
   python convert_hf_to_gguf.py ../qwen-merged --outfile qwen-sparql.gguf --outtype f16
   ```

## 5. Создание и запуск модели в Ollama
1. Скопируйте файл `Modelfile` в папку `llama.cpp`.
2. Создайте модель в Ollama:
   ```bash
   ollama create qwen-sparql -f Modelfile
   ```
3. Запустите модель для работы двумя способами:
   1. **В терминале**
   ```bash
   ollama run qwen-sparql
   ```
   2. **В веб интефрейсе**
   ```bash
   ollama run Web.py
   ```
