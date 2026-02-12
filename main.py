import os
import sys
import argparse
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.ARAGgcn.processing_input import ReviewProcessor
from src.ARAGgcnRetrie.recommender import ARAGgcnRecommender

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
    
    parser.add_argument("--recommender", 
                        choices=['arag', 'araggcn'],
                        default='arag', 
                        help="Recommender")
                        
    parser.add_argument("--input_file", 
                        required=True, 
                        help="Path to the input user history/review JSON file.")
    parser.add_argument("--db-path", 
                        required=False,
                        default = "storage/item_storage" , 
                        help="Path to the FAISS vector database directory.")
                        

    parser.add_argument("--model", 
                        help="Name of the model to use (e.g., 'llama3-8b-8192' for Groq, 'gpt-4o' for OpenAI).")
                        
    parser.add_argument("--embed-model", 
                        default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Name of the sentence-transformer embedding model.")
                        

    args = parser.parse_args()

    load_dotenv()
  
    model = None
    api_key = None
    model_name = args.model

    if args.provider == 'groq':
        api_key_env = 'GROQ_API_KEY'
        if not model_name:
            model_name = 'meta-llama/llama-4-scout-17b-16e-instruct' 
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

    if args.recommender == 'arag' :
        arag_recommender = ARAGRecommender(
            model=model, 
            data_base_path=args.db_path,
            embed_model_name=args.embed_model,
        )
    elif args.recommender == 'araggcn':
        arag_recommender = ARAGgcnRecommender(
            model=model, 
            data_base_path=args.db_path,
            embed_model_name=args.embed_model,
            gcn_model_path=r'./src/ARAGgcn/lgcn/gcn_embeddings_3hop_amazon.pt',
        )
        print("ARAG GCN")
    processor = ReviewProcessor(target_source='amazon')
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
        
        user_reviews_data = combined_data.get("user_reviews")
        candidate_items_data = combined_data.get("candidate_list")

        if not user_reviews_data or not candidate_items_data:
            print("Error: The JSON file must contain both 'user_reviews' and 'candidate_items' keys.")
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{args.input_file}'. Check for formatting errors.")
        sys.exit(1)

    processor.load_reviews(user_reviews_data)

    processor.process_and_split()

    long_term_ctx = processor.long_term_context
    current_session = processor.short_term_context
    " History review user cold start nhiều nhất có 10 cái, ít nhất là 1 "
    " 1 item cho short term, các item còn lại cho long term "
    processor = ReviewProcessor(target_source='amazon')


    if args.recommender == 'arag' :
        final_state = arag_recommender.get_recommendation(
            idx = 0,
            task_set="amazon",
            long_term_ctx=long_term_ctx,
            current_session=current_session,
            nli_threshold=3.0,
            candidate_item = candidate_items_data)
    elif args.recommender == 'araggcn':
        final_state = arag_recommender.get_recommendation(
            idx = 0,
            task_set="amazon",
            user_id ="AE2KV2J6X2OBDKTEAUMIHEXMLFYQ" ,
            long_term_ctx=long_term_ctx,
            current_session=current_session,
            nli_threshold=3.0,
            candidate_item = candidate_items_data)
        print("ARAG GCN")


    
    print("\n\n--- FINAL RANKED LIST ---")
    if final_state.get('final_rank_list'):
        for i, item in enumerate(final_state['final_rank_list']):
            print(f"Rank {i+1}: {item}")

        final_ranker_message = next((msg for msg in reversed(final_state.get('blackboard', [])) if getattr(msg, 'role', '') == "ItemRanker"), None)
        if final_ranker_message and hasattr(final_ranker_message.content, 'explanation'):
            print("\n--- Explanation from Ranker Agent ---")
            print(final_ranker_message.content.explanation)
    else:
        print("No items were ranked.")