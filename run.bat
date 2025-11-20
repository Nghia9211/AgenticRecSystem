@echo off
setlocal

set "PYTHON_SCRIPT=main.py"
set "PROVIDER=groq"
set "INPUT_FILE=tests/test_user.json"
set "DB_PATH=storage/user_storage"

REM SET "MODEL_NAME=llama3-70b-8192"
set "MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct"

set "EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2"

set "DAYS_INTERVAL=20"
set "SHORT_TERM_ITEMS=10"
set "LONG_TERM_ITEMS=50"

echo ======================================================
echo       RUNNING ARAG RECOMMENDER SYSTEM
echo ======================================================
echo Input file:       %INPUT_FILE%
echo Database path:    %DB_PATH%
echo Model:            %MODEL_NAME%
echo ------------------------------------------------------

python %PYTHON_SCRIPT% ^
    --provider "%PROVIDER%" ^
    --input_file "%INPUT_FILE%" ^
    --db-path "%DB_PATH%" ^
    --model "%MODEL_NAME%" ^
    --embed-model "%EMBED_MODEL_NAME%" ^
    --days-i %DAYS_INTERVAL% ^
    --items-k %SHORT_TERM_ITEMS% ^
    --items-m %LONG_TERM_ITEMS%

echo ------------------------------------------------------
echo       SCRIPT EXECUTION FINISHED
echo ======================================================

REM 
pause