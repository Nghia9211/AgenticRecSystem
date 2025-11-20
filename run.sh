#!/bin/bash

set -e

PYTHON_SCRIPT="main.py"

INPUT_FILE="tests/test_user.json"

DB_PATH="storage/user_storage"

#MODEL_NAME="llama3-70b-8192"
MODEL_NAME="gemma-7b-it"
PROVIDER="groq"
EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

DAYS_INTERVAL=20
SHORT_TERM_ITEMS=10
LONG_TERM_ITEMS=50

echo "======================================================"
echo "      RUNNING ARAG RECOMMENDER SYSTEM"
echo "======================================================"
echo "Input file:       $INPUT_FILE"
echo "Database path:    $DB_PATH"
echo "Model:            $MODEL_NAME"
echo "------------------------------------------------------"

python $PYTHON_SCRIPT \
    --provider "$PROVIDER"\
    --input_file "$INPUT_FILE" \
    --db-path "$DB_PATH" \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL_NAME" \
    --days-i $DAYS_INTERVAL \
    --items-k $SHORT_TERM_ITEMS \
    --items-m $LONG_TERM_ITEMS

echo "------------------------------------------------------"
echo "      SCRIPT EXECUTION FINISHED"
echo "======================================================"