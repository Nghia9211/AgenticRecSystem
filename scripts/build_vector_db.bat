@echo off
setlocal


set "MODE=review"

set "PYTHON_SCRIPT=build_vector_db.py"
set "EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2"
set "BATCH_SIZE=256"
set "DATA_DIR=../data"
set "STORAGE_DIR=../storage"

if /I "%MODE%" == "review" (
    echo ^>^>^> Selected Mode : Review ^<^<^<
    echo ======================================================
    echo Building vector store for user reviews...
    echo ======================================================

    python %PYTHON_SCRIPT% ^
        --data_path "%DATA_DIR%/review.json" ^
        --save_path "%STORAGE_DIR%/user_storage" ^
        --embed_model "%EMBED_MODEL%" ^
        --batch_size %BATCH_SIZE%

    echo.
    echo User review database created successfully in '%STORAGE_DIR%/user_storage'.

) else if /I "%MODE%" == "item" (
    echo ^>^>^> Selected Mode : item ^<^<^<
    echo ======================================================
    echo Building vector store for items...
    echo ======================================================

    python %PYTHON_SCRIPT% ^
        --data_path "%DATA_DIR%/item.json" ^
        --save_path "%STORAGE_DIR%/item_storage" ^
        --embed_model "%EMBED_MODEL%" ^
        --batch_size %BATCH_SIZE%

    echo.
    echo Item database created successfully in '%STORAGE_DIR%/item_storage'.
)

echo.
echo ======================================================
echo Script finished.
echo ======================================================

endlocal
pause