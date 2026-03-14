# Урок 1. Играемся с NanoGPT

## 1. Устанавливаем nanoGPT

### 1.1. Создаем новое окружение 
python -m venv nanoGPT_env

nanoGPT_env\Scripts\activate

### 1.2. Устанавливаем PyTorch с поддержкой CUDA (важно для RTX 5090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

### 1.3. Клонируем nanoGPT
git clone https://github.com/karpathy/nanoGPT.git

cd nanoGPT

### 1.4. Устанавливаем зависимости (БЕЗ triton)
pip install torch numpy transformers datasets tiktoken wandb tqdm


### 1.5. Устанавливаем переменные окружения для отключения компиляции
set TORCH_COMPILE_DISABLE=1

set TRITON_DISABLE=1

## 2. Запускаем обучение

### 2.1. Готовим данные 

Объяснение: почему посимвольная, а не по словам? Простота, маленький словарь.

cd data\shakespeare_char

python prepare.py

cd ..\\..

### 2.2. Исправим название директории

в файле config\train_shakespeare_char.py закомментируйте строчку out_dir = 'out-shakespeare-char'

### 2.2. Запускаем обучение
python train.py config\train_shakespeare_char.py
 
## 3. Генерим текст
 python sample.py

## 4. Читаем текст model.py

• Открываем model.py в редакторе.
• Находим класс GPT. Прослеживаем цепочку: GPT → Block → CausalSelfAttention → MLP.
• Объясняем назначение каждого модуля на высоком уровне:
- Embedding – превращает токены в векторы.
- Block – основной строительный блок.
- Attention – механизм обмена информацией между токенами.
- MLP – обработка каждого токена отдельно.
- LayerNorm – нормализация для стабильности.

## 5. Экспериментируем

Меняем параметры в config\train_shakespeare_char.py

n_layers = 6, 2, 12


## 6. Создаем датасет на русском языке

Скачайте 

https://gist.githubusercontent.com/romaklimenko/c95f3a864828f7f034b7a33d1676e420/raw/55f9027799b5b3c67e2f7cb3d6a7154f707ff08a/warandpeace.txt

и с помощью prepare.py создайте датасет.

## 7. Обучаем модель на русском датасете

Обучите модель и посмотрите на первые результаты. 

## 8. Исследуем температуру


Поменяйте в model.py и посмотрите на результаты. При какой температуре сеть больше похожа на человека?

temperature=0.1, 1.0, 1.5, 5.0
