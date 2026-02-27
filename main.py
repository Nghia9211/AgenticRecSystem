import os
import sys
import argparse
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.ARAGgcn.processing_input import ReviewProcessor

from src.ARAG.recommender import ARAGRecommender
from src.ARAG_init.recommender import ARAGrawRecommender
from src.ARAGgcnRetrie.recommender import ARAGgcnRetrieRecommender
from src.ARAGgcn.recommender import ARAGgcnRecommender
from src.ARAGfinal.recommender import ARAGgcnv2Recommender

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ARAG Recommender System.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    
    # Thêm lựa chọn dataset
    parser.add_argument("--dataset", 
                        choices=['amazon', 'goodreads', 'yelp'],
                        default='amazon', 
                        help="Choose the dataset to work with.")

    parser.add_argument("--provider", 
                        choices=['groq', 'openai'],
                        default='groq', 
                        help="The LLM provider to use.")
    
    parser.add_argument("--recommender", 
                        choices=['arag', 'araggcn', 'araginit', 'araggcnretrie', 'aragv2'],
                        default='arag', 
                        help="Recommender type")
                        
    parser.add_argument("--model", 
                        help="Name of the model to use.")
                        
    parser.add_argument("--embed-model", 
                        default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Name of the sentence-transformer embedding model.")

    args = parser.parse_args()

    # --- CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG THEO DATASET ---
    DATASET_CONFIG = {
        "amazon": {
            "input_file": "tests/user_amazon.json",
            "db_path": "storage/item_storage_amazon",
            "gcn_pt": r'./src/ARAGgcn/lgcn/gcn_embeddings_3hop_amazon.pt',
            "mask_file": "data/graph_data/final_mask_amazon.json"
        },
        "goodreads": {
            "input_file": "tests/user_goodreads.json",
            "db_path": "storage/item_storage_goodreads",
            "gcn_pt": r'./src/ARAGgcn/lgcn/gcn_embeddings_3hop_goodreads.pt',
            "mask_file": "data/graph_data/final_mask_goodreads.json"
        },
        "yelp": {
            "input_file": "tests/user_yelp.json",
            "db_path": "storage/item_storage_yelp",
            "gcn_pt": r'./src/ARAGgcn/lgcn/gcn_embeddings_3hop_yelp.pt',
            "mask_file": "data/graph_data/final_mask_yelp.json"
        }
    }

    config = DATASET_CONFIG[args.dataset]
    input_file = config["input_file"]
    db_path = config["db_path"]
    gcn_model_path = config["gcn_pt"]

    load_dotenv()
  
    model = None
    model_name = args.model

    # --- LLM Provider Setup ---
    if args.provider == 'groq':
        if not model_name: model_name = 'meta-llama/llama-4-scout-17b-16e-instruct' 
        api_key = os.getenv('GROQ_API_KEY')
        model = ChatGroq(model=model_name, api_key=api_key)
    elif args.provider == 'openai':
        if not model_name: model_name = 'gpt-3.5-turbo'
        api_key = os.getenv('OPENAI_API_KEY')
        model = ChatOpenAI(model=model_name, api_key=api_key)

    # --- Recommender Initialization ---
    if args.recommender == 'arag':
        arag_recommender = ARAGRecommender(
            model=model, 
            data_base_path=db_path,
            embed_model_name=args.embed_model,
        )
    elif args.recommender == 'araggcn':
        arag_recommender = ARAGgcnRecommender(
            model=model, 
            data_base_path=db_path,
            embed_model_name=args.embed_model,
            gcn_model_path=gcn_model_path,
        )
    elif args.recommender == 'araginit':
        arag_recommender = ARAGrawRecommender(
            model=model, 
            data_base_path=db_path,
            embed_model_name=args.embed_model,
        )
    elif args.recommender == 'araggcnretrie':
        arag_recommender = ARAGgcnRetrieRecommender(
            model=model, 
            data_base_path=db_path,
            embed_model_name=args.embed_model,
            gcn_model_path=gcn_model_path,
        )
    elif args.recommender == 'aragv2':
        arag_recommender = ARAGgcnv2Recommender(
                model=model,
                data_base_path=db_path,
                gcn_embedding_path=gcn_model_path,
                gcn_review_file=f"data/review.json", 
                gcn_item_file=f"data/item.json",
                gcn_user_file=f"data/user.json",
                gcn_gt_mask_file=config["mask_file"],
            )

    # --- Processing Input ---
    processor = ReviewProcessor(target_source=args.dataset)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
        
        user_reviews_data = combined_data.get("user_reviews")
        candidate_items_data = combined_data.get("candidate_list")
        user_id = combined_data.get("user_id", "DEFAULT_USER") 

        if not user_reviews_data or not candidate_items_data:
            print(f"Error: JSON file {input_file} missing keys.")
            sys.exit(1)

    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    processor.load_reviews(user_reviews_data)
    processor.process_and_split()

    long_term_ctx = processor.long_term_context
    current_session = processor.short_term_context

    # --- Get Recommendation ---
    # Thay task_set="amazon" bằng args.dataset
    recommendation_params = {
        "idx": 0,
        "task_set": args.dataset,
        "long_term_ctx": long_term_ctx,
        "current_session": current_session,
        "nli_threshold": 3.0,
        "candidate_item": candidate_items_data
    }

    # Một số model cần user_id
    if args.recommender in ['araggcn', 'araggcnretrie', 'aragv2']:
        recommendation_params["user_id"] = user_id
    if args.recommender == 'aragv2':
        recommendation_params["gcn_top_K"] = 20

    final_state = arag_recommender.get_recommendation(**recommendation_params)

    # --- Output ---
    print(f"\n\n--- FINAL RANKED LIST ({args.dataset.upper()}) ---")
    if final_state.get('final_rank_list'):
        for i, item in enumerate(final_state['final_rank_list']):
            print(f"Rank {i+1}: {item}")
        
        # Tìm tin nhắn cuối cùng của ItemRanker để in giải thích
        blackboard = final_state.get('blackboard', [])
        for msg in reversed(blackboard):
            if getattr(msg, 'role', '') == "ItemRanker":
                if hasattr(msg.content, 'explanation'):
                    print("\n--- Explanation from Ranker Agent ---")
                    print(msg.content.explanation)
                break
    else:
        print("No items were ranked.")