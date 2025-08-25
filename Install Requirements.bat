@echo off

echo Requirements
echo.


rem "C:\Python312\python" -m pip install -r requirements.txt
python310\App\Python\python -m pip install --upgrade pip --no-warn-script-location
python310\App\Python\python -m pip install --upgrade transformers tokenizers safetensors langdetect langid huggingface_hub --no-warn-script-location
python310\App\Python\python -m pip install --upgrade --force-reinstall sentencepiece==0.2.0 --no-warn-script-location
python310\App\Python\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-warn-script-location



@REM pause