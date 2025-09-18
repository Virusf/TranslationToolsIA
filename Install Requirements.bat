@echo off

echo Requirements
echo.


rem "C:\Python312\python" -m pip install -r requirements.txt
python310\App\Python\python -m pip install --upgrade pip --no-warn-script-location
python310\App\Python\python -m pip install --upgrade transformers tokenizers safetensors langdetect langid huggingface_hub hf_xet peft --no-warn-script-location
python310\App\Python\python -m pip install --upgrade torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location
python310\App\Python\python -m pip install --upgrade --force-reinstall sentencepiece==0.2.0 --no-warn-script-location




@REM pause