# Diplom-25-26
1. Требуется скачать и разархивировать папку 
2. Открыть папку в VSC
3. Установить зависимости введя в терминал:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install peft==0.7.0
pip install bitsandbytes==0.41.3
pip install accelerate==0.25.0
4. Запустить Phy.py
5. Запустить convert_to_ollama.py
6. Устанавливаем llama.cpp для конвервирования моделив GGUF
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
7. Переходим в папку установленной llama и запускаем конвертацию с помощью скрипта ниже
cd C:\Users\User\Desktop\govno\sparql-model\llama.cpp
python convert_hf_to_gguf.py ..\phi2-sparql-merged --outfile phi2-sparql.gguf --outtype q8_0
8. Скопируйте Modelfile в папку llama.cpp
9. Создайте модель введя
ollama create sparql-assistant -f Modelfile
10. Запустите модель и вводите вопросы
ollama run sparql-assistant
