@echo off
TITLE Build Graph Data for Recommendation System
COLOR 0A

set PYTHON_SCRIPT=build_graph_gcn.py

set TASK_TYPE="goodreads"

set USER_FILE="C:\Users\Admin\Desktop\Document\SpeechToText\RecSystemCode\data\graph_data\user_%TASK_TYPE%.json"
set ITEM_FILE="C:\Users\Admin\Desktop\Document\SpeechToText\RecSystemCode\data\graph_data\item_%TASK_TYPE%.json"
set REVIEW_FILE="C:\Users\Admin\Desktop\Document\SpeechToText\RecSystemCode\data\graph_data\review_%TASK_TYPE%.json"

set OUTPUT_FILE="processed_graph_data_%TASK_TYPE%.pt"



echo ========================================================
echo        STARTING GRAPH BUILD PROCESS
echo ========================================================
echo Script: %PYTHON_SCRIPT%
echo.

python %PYTHON_SCRIPT% --user_file %USER_FILE% --item_file %ITEM_FILE% --review_file %REVIEW_FILE% --output_file %OUTPUT_FILE% 

echo.
echo ========================================================
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Graph data built successfully!
) else (
    echo [ERROR] Something went wrong. Check the logs above.
)
echo ========================================================


set DATA_FILE=processed_graph_data_%TASK_TYPE%.pt
set EXPORT_FILE=gcn_embeddings_3hop_%TASK_TYPE%.pt
set EPOCHS=1000
set BATCH_SIZE=1024
set LR=0.01
set REG=1e-4
set EMB_DIM=64
set DEVICE=cuda

echo Running training with:
echo - Epochs: %EPOCHS%
echo - Batch Size: %BATCH_SIZE%
echo - Learning Rate: %LR%
echo - Device: %DEVICE%
echo.

python train.py ^
    --data_file "%DATA_FILE%" ^
    --export_file "%EXPORT_FILE%" ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --reg %REG% ^
    --emb_dim %EMBED_DIM% ^
    --device %DEVICE%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed or interrupted!
    pause
    exit /b
)

echo.
echo ==================================================
echo           TRAINING COMPLETED SUCCESSFULLY
echo ==================================================
pause