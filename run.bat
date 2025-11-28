@echo off
setlocal

set "PYTHON_SCRIPT=main.py"
set "PROVIDER=groq"
set "INPUT_FILE=tests/user_amazon.json"
set "DB_PATH=storage/user_storage"
set "RECOMMENDER=araggcn"
set "MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct"

set "EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2"


echo ======================================================
echo       RUNNING ARAG RECOMMENDER SYSTEM
echo ======================================================
echo Input file:       %INPUT_FILE%
echo Database path:    %DB_PATH%
echo Model:            %MODEL_NAME%
echo ------------------------------------------------------

python %PYTHON_SCRIPT% ^
    --provider "%PROVIDER%" ^
    --recommender "%RECOMMENDER%"^
    --input_file "%INPUT_FILE%" ^
    --db-path "%DB_PATH%" ^
    --model "%MODEL_NAME%" ^
    --embed-model "%EMBED_MODEL_NAME%" 

echo ------------------------------------------------------
echo       SCRIPT EXECUTION FINISHED
echo ======================================================

REM 
pause