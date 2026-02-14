import json
from typing import Optional

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
import ast
from .prompts import *
from .schemas import (BlackboardMessage,
                          RankedItem, RecState)
from .metric import evaluate_hit_rate
from .utils import *

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CHANGE: Import GCN module                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝
from .gcn_module import GCNEngine


class ARAGAgents:
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  CHANGE: Accept optional gcn_engine parameter                    ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    def __init__(self, model, score_model, rank_model, embedding_function,
                 gcn_engine: Optional[GCNEngine] = None):
        self.model = model
        self.score_model = score_model
        self.rank_model = rank_model
        self.embedding_function = embedding_function
        self.gcn_engine = gcn_engine  # ← NEW: GCN engine instance

    def _get_gt_path(self, state):
        return f"C:/Users/Admin/Desktop/Document/AgenticCode/AgentRecBench/dataset/task/user_cold_start/{state['task_set']}/groundtruth"

    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  NEW NODE: GCN Pre-Processing (Strategy 1 + Strategy 2)          ║
    # ║                                                                  ║
    # ║  This runs BEFORE everything else. It:                           ║
    # ║    1) Expands user context with neighbor items (Strategy 1)      ║
    # ║    2) Pre-filters candidate corpus (Strategy 2)                  ║
    # ║                                                                  ║
    # ║  Both outputs are written to state for downstream agents.        ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    def gcn_preprocess(self, state: RecState):
        """
        GCN Pre-Processing Node — runs at the START of the pipeline.
        
        Strategy 1: Finds neighbor users via GCN, collects their items 
                    as H_expanded → feeds into UUA for richer understanding
        
        Strategy 2: Scores all candidates via GCN user-item similarity,
                    keeps top-K → narrows the corpus for RAG retrieval
        """
        user_id = state.get('user_id', 'unknown')
        candidate_list = state['candidate_list']
        
        gcn_expanded_context = []
        gcn_filtered_candidates = candidate_list  # default: no filtering
        
        if self.gcn_engine and self.gcn_engine.is_trained:
            # --- Strategy 1: Neighbor Context Expansion ---
            # Extract item IDs from user's history to exclude from expansion
            user_history_ids = self._extract_history_item_ids(state)
            
            gcn_expanded_context = self.gcn_engine.get_expanded_context(
                user_id=user_id,
                user_history_item_ids=user_history_ids,
                top_m_neighbors=5,      # find 5 similar users
                max_expanded_items=10   # take up to 10 items from them
            )
            
            # --- Strategy 2: Corpus Pre-Filtering ---
            gcn_top_K = state.get('gcn_top_K', 50)  # configurable
            gcn_filtered_candidates = self.gcn_engine.prefilter_candidates(
                user_id=user_id,
                candidate_list=candidate_list,
                top_K=gcn_top_K
            )
            
            print(f"\n{'='*50}")
            print(f"🔬 [GCN PRE-PROCESS] Strategy 1: {len(gcn_expanded_context)} expanded items")
            print(f"🔬 [GCN PRE-PROCESS] Strategy 2: {len(candidate_list)} → {len(gcn_filtered_candidates)} candidates")
            print(f"{'='*50}\n")
        else:
            print("[GCN PRE-PROCESS] No trained GCN engine. Using defaults (no expansion, no filtering).")

        # Write results to state
        return {
            'gcn_expanded_context': gcn_expanded_context,
            'gcn_filtered_candidates': gcn_filtered_candidates
        }
    
    def _extract_history_item_ids(self, state: RecState) -> List[str]:
        """Helper: extract item IDs from long_term_ctx for GCN exclusion."""
        ids = []
        try:
            lt_ctx = state.get('long_term_ctx', '')
            if isinstance(lt_ctx, str):
                # Try parsing as JSON list
                items = json.loads(lt_ctx) if lt_ctx.startswith('[') else []
                for item in items:
                    if isinstance(item, dict):
                        ids.append(str(item.get('item_id', item.get('sub_item_id', ''))))
        except Exception:
            pass
        return ids

    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  MODIFIED: User Understanding Agent now receives GCN expansion   ║
    # ║                                                                  ║
    # ║  KEY CHANGE: If gcn_expanded_context exists in state, it's       ║
    # ║  formatted and passed to UUA prompt so the agent can infer       ║
    # ║  latent preferences from neighbor behavior.                      ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    def user_understanding_agent(self, state: RecState):
        lt_ctx = state['long_term_ctx']
        cur_ses = state['current_session']
        
        # ── CHANGE: Check for GCN expanded context ──
        gcn_expanded = state.get('gcn_expanded_context', [])
        
        if gcn_expanded:
            # Format expanded context as readable text for the LLM
            gcn_context_str = self._format_gcn_expanded_context(gcn_expanded)
            
            # Use the GCN-enhanced prompt (Strategy 1)
            prompt = create_summary_user_behavior_with_gcn_prompt(
                lt_ctx=lt_ctx,
                cur_ses=cur_ses,
                gcn_behavior_insight=gcn_context_str
            )
            print(f"[UUA] Using GCN-enhanced prompt with {len(gcn_expanded)} neighbor items")
        else:
            # Fallback to original prompt (no GCN)
            prompt = create_summary_user_behavior_prompt(lt_ctx=lt_ctx, cur_ses=cur_ses)
            print("[UUA] Using standard prompt (no GCN context)")
        
        response = self.model.invoke(prompt)
        uua_output = response.content

        uua_blackboard_message = BlackboardMessage(
            role="UserUnderStanding",
            content=uua_output
        )

        return {"blackboard": [uua_blackboard_message]}
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  NEW HELPER: Format GCN expanded items for LLM consumption      ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    def _format_gcn_expanded_context(self, gcn_items: list) -> str:
        """
        Convert GCN neighbor items into a readable format for the UUA prompt.
        
        IMPORTANT: We label these clearly as "similar user signals" so the 
        LLM treats them as WEAK evidence of latent preferences, not as 
        direct user history. This prevents the UUA from confusing neighbor 
        behavior with actual user behavior.
        """
        lines = []
        for i, item in enumerate(gcn_items, 1):
            title = item.get('title', 'Unknown')
            desc = item.get('description', '')[:150]
            pop = item.get('neighbor_popularity', 0)
            lines.append(
                f"  {i}. \"{title}\" — {desc}"
                f" [Liked by {pop} similar user(s)]"
            )
        
        return (
            "The following items were NOT in this user's history, but were "
            "frequently interacted with by users who have very similar "
            "behavioral patterns (discovered via graph analysis):\n"
            + "\n".join(lines)
        )

    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  MODIFIED: initial_retrieval now uses GCN-filtered candidates    ║
    # ║                                                                  ║
    # ║  KEY CHANGE: Instead of searching the full candidate_list,       ║
    # ║  we search gcn_filtered_candidates (Strategy 2 output).          ║
    # ║  This means RAG operates on a pre-narrowed, higher-quality       ║
    # ║  corpus where the base rate of relevant items is much higher.    ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    def initial_retrieval(self, state: RecState):
        user_understanding_msg = get_user_understanding(state)
        
        # ── CHANGE: Use GCN-filtered candidates if available ──
        # Strategy 2: RAG searches a narrowed corpus, not the full list
        candidate_list = state.get('gcn_filtered_candidates', state['candidate_list'])
        
        print(f"[RAG Retrieval] Searching {len(candidate_list)} candidates "
              f"(original: {len(state['candidate_list'])})")

        query = f' User Preference : {user_understanding_msg} \n '

        top_k_list = find_top_k_similar_items(query, candidate_list, self.embedding_function)

        evaluate_hit_rate(
            index=state['idx'],
            stage="1_Initial_Retrieval",
            items=top_k_list,
            gt_folder=self._get_gt_path(state),
            task_set=state['task_set']
        )
        return {'top_k_candidate': top_k_list}

    # ─────────────────────────────────────────────────────────────────────
    # NLI Agent — UNCHANGED (no modifications needed)
    # ─────────────────────────────────────────────────────────────────────
    def nli_agent(self, state: RecState, config: Optional[RunnableConfig] = None):
        threshold = config.get("configurable", {}).get("nli_threshold", 5.5) if config else 5.5
        top_k_candidate_raw = state['top_k_candidate']
        user_understanding_msg = get_user_understanding(state)

        if not top_k_candidate_raw:
            return {'positive_list': [], "blackboard": []}

        top_k_candidate = []
        for item in top_k_candidate_raw:
            top_k_candidate.append(normalize_item_data(item))

        prompts_list = [
            create_assess_nli_score_prompt2(
                item=item, user_preferences=user_understanding_msg, item_id=item['item_id'])
            for item in top_k_candidate
        ]
        all_nli_outputs = self.score_model.batch(prompts_list)

        positive_item_list = []
        messages = []

        print(f"\033[93m[NLI Scoring]\033[0m Threshold: {threshold}")
        for item, nli_output in zip(top_k_candidate, all_nli_outputs):
            item_name = item.get('name') or item.get('title') or "Unknown"

            status = "✅ PASS" if nli_output.score >= threshold else "❌ FAIL"
            print(f"  - {status} | Score: {nli_output.score:.1f} | Item: {item_name}")
            print(f"---")
            print(f"Item: {item_name} | Score: {nli_output.score}")
            print(f"Rationale: {nli_output.rationale}")
            if nli_output.score >= threshold:
                positive_item_list.append(item)

            messages.append(
                BlackboardMessage(
                    role="NaturalLanguageInference",
                    content=nli_output,
                    score=nli_output.score
                )
            )

        evaluate_hit_rate(
            index=state['idx'],
            stage="2_NLI_Filtering",
            items=positive_item_list,
            gt_folder=self._get_gt_path(state),
            task_set=state['task_set']
        )

        return {'positive_list': positive_item_list, "blackboard": messages}

    # ─────────────────────────────────────────────────────────────────────
    # Context Summary Agent — UNCHANGED
    # ─────────────────────────────────────────────────────────────────────
    def context_summary_agent(self, state: RecState):
        blackboard = state['blackboard']
        positive_item = state['positive_list']

        if not positive_item:
            print("No positive items to summarize. Skipping.")
            return {"blackboard": [BlackboardMessage(role="ContextSummary", content="No positive items were found to summarize.")]}

        user_understanding_msg = get_user_understanding(state)
        nli_messages = [msg for msg in blackboard if msg.role == "NaturalLanguageInference"]
        positive_item_ids = {item['item_id'] for item in positive_item}

        items_with_scores_str = ""
        for msg in nli_messages:
            if msg.content.item_id in positive_item_ids:
                item_data = next(
                    (item for item in positive_item if item['item_id'] == msg.content.item_id), None)
                if item_data:
                    items_with_scores_str += (
                        f"Item: {item_data}\n"
                        f"NLI Score: {msg.score}/10\n"
                        f"Rationale: {msg.content.rationale}\n---\n"
                    )

        prompt = create_context_summary_prompt(
            user_summary=user_understanding_msg, items_with_scores_str=items_with_scores_str)
        response = self.model.invoke(prompt)
        csa_output = response.content

        csa_blackboard_message = BlackboardMessage(
            role="ContextSummary",
            content=csa_output
        )

        return {'blackboard': [csa_blackboard_message]}

    # ─────────────────────────────────────────────────────────────────────
    # Item Ranker Agent — UNCHANGED
    # ─────────────────────────────────────────────────────────────────────
    def item_ranker_agent(self, state: RecState):
        print("\n--- [DEBUG ITEM RANKER] ---")
        items_to_rank = state['positive_list']
        candidate_list = state['candidate_list']

        if not items_to_rank:
            print("⚠️ [DEBUG] No items in positive_list. Skipping LLM call.")
            final_list = [item.get('item_id') for item in candidate_list]
            return {'final_rank_list': final_list}

        context_summary = get_user_summary(state)
        user_understanding = get_user_understanding(state)

        print(f"✅ [DEBUG] Items to rank count: {len(items_to_rank)}")
        print(f"✅ [DEBUG] User Summary Length: {len(user_understanding)}")

        items_to_rank_str = json.dumps(items_to_rank, indent=2, ensure_ascii=False)

        prompt = create_item_ranking_prompt(
            user_summary=user_understanding,
            context_summary=context_summary,
            items_to_rank=items_to_rank_str
        )
        result_from_model = None

        try:
            print("🚀 [DEBUG] Sending request to Groq Model...")
            result_from_model = self.rank_model.invoke(prompt)
            print(f"✅ [DEBUG] Model Response Received. Explanation len: {len(result_from_model.explanation if result_from_model else '')}")
        except Exception as e:
            print(f"❌ [DEBUG] LỖI KHI GỌI MODEL RANKER: {str(e)}")
            import traceback
            traceback.print_exc()

        if not result_from_model:
            print("⚠️ [DEBUG] Model failed/Returned None. Using fallback order.")
            ranked_positive_items = [
                RankedItem(
                    item_id=str(i.get('item_id')),
                    name=str(i.get('title') or i.get('name') or 'Unknown'),
                    description="Fallback"
                ) for i in items_to_rank
            ]
        else:
            ranked_positive_items = result_from_model.ranked_list

        ranked_item_ids = {str(item.item_id) for item in ranked_positive_items}
        unranked_items_ids = [str(item['item_id']) for item in candidate_list if str(item['item_id']) not in ranked_item_ids]

        final_result_ids = [str(item.item_id) for item in ranked_positive_items] + unranked_items_ids

        print(f"🏆 [DEBUG] Final Rank Order: {final_result_ids[:5]}... (Total: {len(final_result_ids)})")

        item_ranking_message = BlackboardMessage(
            role="ItemRanker",
            content=result_from_model if result_from_model else "Fallback ranking used"
        )

        return {'final_rank_list': final_result_ids, 'blackboard': [item_ranking_message]}

    # ─────────────────────────────────────────────────────────────────────
    # Conditional edge — UNCHANGED
    # ─────────────────────────────────────────────────────────────────────
    def should_proceed_to_summary(self, state: RecState):
        if not state.get('positive_list') or len(state['positive_list']) == 0:
            print("No positive items found after NLI. Stopping.")
            return END
        return "continue"