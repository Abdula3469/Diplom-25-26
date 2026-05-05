# Инструкция запуска

Этот проект посвящен объединению и запуску модели для генерации SPARQL-запросов с использованием Ollama.

## 1. Подготовка
1. Скачайте и разархивируйте папку с проектом.
2. Откройте разархивированную папку в **Visual Studio Code (VSC)**.

## 2. Установка зависимостей
Откройте терминал в VSC и выполните следующую команду для установки необходимых библиотек:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0 peft==0.7.0 bitsandbytes==0.41.3 accelerate==0.25.0
```

## 3. Обучение и подготовка модели
Запустите последовательно следующие скрипты:
1. **Phy.py** (обучение модели):
   ```bash
   python Phy.py
   ```
2. **run_phy** (Запуск модели):
   ```bash
   python convert_to_ollama.py
   ```

## 4. Конвертация в формат GGUF(пока что устарело)
Для работы с Ollama модель нужно конвертировать с помощью `llama.cpp`.

1. Склонируйте репозиторий `llama.cpp`:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   ```
2. Запустите конвертацию (убедитесь, что путь к модели верный):
   ```bash
   python convert_hf_to_gguf.py ..\phi2-sparql-merged --outfile phi2-sparql.gguf --outtype q8_0
   ```

## 5. Создание и запуск модели в Ollama
1. Скопируйте файл `Modelfile` в папку `llama.cpp`.
2. Создайте модель в Ollama:
   ```bash
   ollama create sparql-assistant -f Modelfile
   ```
3. Запустите модель для работы:
   ```bash
   ollama run sparql-assistant
   ```
