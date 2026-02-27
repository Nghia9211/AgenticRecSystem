@echo off
setlocal

set "PYTHON_SCRIPT=main.py"
set "PROVIDER=groq"

:: CHỈ CẦN THAY ĐỔI DATASET Ở ĐÂY: amazon, goodreads, hoặc yelp
set "DATASET=yelp"

set "RECOMMENDER=arag"
set "MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct"
set "EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2"

echo ======================================================
echo       RUNNING ARAG: %DATASET%
echo ======================================================

python %PYTHON_SCRIPT% ^
    --dataset "%DATASET%" ^
    --provider "%PROVIDER%" ^
    --recommender "%RECOMMENDER%" ^
    --model "%MODEL_NAME%" ^
    --embed-model "%EMBED_MODEL_NAME%" 

echo ------------------------------------------------------
pause