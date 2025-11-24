import os
import sys
import argparse
from dotenv import load_dotenv

# Nhập cả hai lớp mô hình từ LangChain
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.ARAG.processing_input import ReviewProcessor
from src.ARAG.recommender import ARAGRecommender

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ARAG Recommender System.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    
    parser.add_argument("--provider", 
                        choices=['groq', 'openai'],
                        default='groq', 
                        help="The LLM provider to use.")
                        
    parser.add_argument("--input_file", 
                        required=True, 
                        help="Path to the input user history/review JSON file.")
    parser.add_argument("--db-path", 
                        required=False,
                        default = "storage/user_storage" , 
                        help="Path to the FAISS vector database directory.")
                        

    parser.add_argument("--model", 
                        help="Name of the model to use (e.g., 'llama3-8b-8192' for Groq, 'gpt-4o' for OpenAI).")
                        
    parser.add_argument("--embed-model", 
                        default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Name of the sentence-transformer embedding model.")
                        
    
    parser.add_argument("--days-i", 
                        type=int, 
                        default=20, 
                        help="Maximum number of days for Short Term Context.")
    parser.add_argument("--items-k", 
                        type=int, 
                        default=10, 
                        help="Maximum number of items for Short Term Context.")
    parser.add_argument("--items-m", 
                        type=int, 
                        default=50, 
                        help="Maximum number of items for Long Term Context.")
    args = parser.parse_args()

    load_dotenv()
  
    model = None
    api_key = None
    model_name = args.model

    if args.provider == 'groq':
        api_key_env = 'GROQ_API_KEY'
        if not model_name:
            model_name = 'llama3-8b-8192' 
        api_key = os.getenv(api_key_env)
        if not api_key:
            print(f"Error: Environment variable '{api_key_env}' not found for the 'groq' provider.")
            sys.exit(1)
        print(f"Using Groq provider with model: {model_name}")
        model = ChatGroq(model=model_name, api_key=api_key)

    elif args.provider == 'openai':
        api_key_env = 'OPENAI_API_KEY'
        if not model_name:
            model_name = 'gpt-3.5-turbo'
        api_key = os.getenv(api_key_env)
        if not api_key:
            print(f"Error: Environment variable '{api_key_env}' not found for the 'openai' provider.")
            sys.exit(1)
        print(f"Using OpenAI provider with model: {model_name}")
        model = ChatOpenAI(model=model_name, api_key=api_key)

    else:
        print(f"Error: Invalid provider '{args.provider}'. Please choose 'groq' or 'openai'.")
        sys.exit(1)

    arag_recommender = ARAGRecommender(
        model=model, 
        data_base_path=args.db_path,
        embed_model_name=args.embed_model
    )
    
    processor = ReviewProcessor()
    
    try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                combined_data = json.load(f)
            
            user_reviews_data = combined_data.get("user_reviews")
            candidate_items_data = combined_data.get("candidate_items")

            if not user_reviews_data or not candidate_items_data:
                print("Error: The JSON file must contain both 'user_reviews' and 'candidate_items' keys.")
                sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{args.input_file}'. Check for formatting errors.")
        sys.exit(1)

    if not processor.load_reviews(user_reviews_data):
        print("Could not load reviews from the input file.")
        sys.exit(1)

    processor.process_and_split(args.days_i, args.items_k, args.items_m)

    long_term_ctx = processor.long_term_context
    current_session = processor.short_term_context

    final_state = arag_recommender.get_recommendation(
        long_term_ctx=long_term_ctx,
        current_session=current_session,
        nli_threshold=3.0,
        candidate_item = candidate_items_data)
    
    print("\n\n--- FINAL RANKED LIST ---")
    if final_state.get('final_rank_list'):
        for i, item in enumerate(final_state['final_rank_list']):
            print(f"Rank {i+1}: {getattr(item, 'item_id', 'Unknown ID')}")

        final_ranker_message = next((msg for msg in reversed(final_state.get('blackboard', [])) if getattr(msg, 'role', '') == "ItemRanker"), None)
        if final_ranker_message and hasattr(final_ranker_message.content, 'explanation'):
            print("\n--- Explanation from Ranker Agent ---")
            print(final_ranker_message.content.explanation)
    else:
        print("No items were ranked.")