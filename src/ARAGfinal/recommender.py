from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from typing import Optional

from .agents import ARAGAgents
from .graph_builder import GraphBuilder
from .schemas import ItemRankerContent, NLIContent, RecState
from .gcn_module import GCNEngine


class ARAGgcnv2Recommender:
    """
    ARAG Recommender with optional GCN Strategy 1+2 integration.
    
    GCN Loading — TWO options:
      Option A: Pass a pre-built GCNEngine instance
      Option B: Pass file paths and let the recommender build the engine
    
    If no GCN is provided, runs standard ARAG (fully backward-compatible).
    """
    
    def __init__(
        self,
        model: ChatGroq,
        data_base_path: str,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        # ╔═══════════════════════════════════════════════════════════════╗
        # ║  GCN Integration Options                                     ║
        # ║                                                              ║
        # ║  Option A: pass a ready GCNEngine                            ║
        # ║  Option B: pass file paths to your LightGCN outputs          ║
        # ╚═══════════════════════════════════════════════════════════════╝
        gcn_engine: Optional[GCNEngine] = None,
        gcn_embedding_path: Optional[str] = None,  # gcn_embeddings_3hop_*.pt
        gcn_review_file: Optional[str] = None,      # review_*.json
        gcn_item_file: Optional[str] = None,         # item_*.json
        gcn_user_file: Optional[str] = None,         # user_*.json
        gcn_gt_mask_file: Optional[str] = None,      # final_mask_*.json
    ):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embed_model_name
        )

        self.loaded_vector_store = FAISS.load_local(
            folder_path=data_base_path,
            embeddings=self.embedding_function,
            allow_dangerous_deserialization=True,
            distance_strategy="COSINE"
        )

        # ── Initialize GCN Engine ──
        if gcn_engine:
            # Option A: pre-built engine
            self.gcn_engine = gcn_engine
        elif gcn_embedding_path and gcn_review_file:
            # Option B: build from your LightGCN output files
            self.gcn_engine = GCNEngine()
            self.gcn_engine.load(
                embedding_path=gcn_embedding_path,
                review_file=gcn_review_file,
                item_file=gcn_item_file,
                user_file=gcn_user_file,
                gt_mask_file=gcn_gt_mask_file
            )
        else:
            self.gcn_engine = None
            print("[ARAG] No GCN provided. Running standard ARAG.")

        self.agents = ARAGAgents(
            model=model,
            score_model=model.with_structured_output(NLIContent),
            rank_model=model.with_structured_output(ItemRankerContent),
            embedding_function=self.embedding_function,
            gcn_engine=self.gcn_engine
        )
        builder = GraphBuilder(agent_provider=self.agents)
        self.workflow = builder.build()

    def get_recommendation(
        self,
        idx: int,
        task_set: str,
        long_term_ctx: str,
        current_session: str,
        candidate_item: dict,
        nli_threshold: float = 4.0,
        user_id: str = "unknown",
        gcn_top_K: int = 50
    ) -> RecState:
        print("\n" + "=" * 50)
        print("🚀 [START] ARAG RECOMMENDATION ENGINE")
        if self.gcn_engine and self.gcn_engine.is_trained:
            print(f"   📊 GCN Strategy 1+2 ACTIVE (user={user_id}, K={gcn_top_K})")
        else:
            print("   ⚡ Standard ARAG (no GCN)")
        print("=" * 50)

        run_config = {"configurable": {"nli_threshold": nli_threshold}}

        initial_state = {
            "idx": idx,
            "task_set": task_set,
            "long_term_ctx": long_term_ctx,
            "current_session": current_session,
            "blackboard": [],
            "candidate_list": candidate_item,
            "user_id": user_id,
            "gcn_top_K": gcn_top_K,
            "gcn_expanded_context": [],
            "gcn_filtered_candidates": [],
        }
        final_state = self.workflow.invoke(initial_state, config=run_config)
        return final_state