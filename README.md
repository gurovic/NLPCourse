# Урок 1. Играемся с NanoGPT

## 1. Создаем новое окружение 
python -m venv nanoGPT_env
nanoGPT_env\Scripts\activate

## 2. Устанавливаем PyTorch с поддержкой CUDA (важно для RTX 5090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

## 3. Клонируем nanoGPT
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

## 4. Устанавливаем зависимости (БЕЗ triton)
pip install torch numpy transformers datasets tiktoken wandb tqdm


## 5. Устанавливаем переменные окружения для отключения компиляции
set TORCH_COMPILE_DISABLE=1
set TRITON_DISABLE=1

## 6. Готовим данные
cd data\shakespeare_char
python prepare.py
cd ..\..

## 7. Запускаем обучение
python train.py config\train_shakespeare_char.py
