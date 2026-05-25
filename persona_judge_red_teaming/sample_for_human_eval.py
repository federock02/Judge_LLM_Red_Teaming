import os
import json
import random
import argparse
import uuid
from typing import List

# Import your existing utilities (adjust these imports based on your project structure)
from analyze_results import load_edges, EmbeddingCache # Assuming you use this for similarity

def main():
    parser = argparse.ArgumentParser(description="Sample successful prompts for blind human evaluation.")
    parser.add_argument("--results_folders", type=str, nargs="+", required=True,
                        help="Folders containing red_teaming_archive.json")
    parser.add_argument("--out_dir", type=str, default="human_evaluation",
                        help="Directory to save the blind and key JSONs")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of successful prompts to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Data
    archive_paths = []
    for folder in args.results_folders:
        path = os.path.abspath(os.path.join(folder, "red_teaming_archive.json"))
        if os.path.isfile(path):
            archive_paths.append(path)
            
    if not archive_paths:
        print("[ERROR] No valid archives found.")
        return

    print(f"[INFO] Loading edges from {len(archive_paths)} archives...")
    all_edges = load_edges(archive_paths)

    # 2. Filter for successful attacks and deduplicate
    print("[INFO] Filtering for successful, unique mutations...")
    seen_children = set()
    unique_successful_edges = []
    
    for edge in all_edges:
        if edge.success and edge.child_prompt not in seen_children:
            seen_children.add(edge.child_prompt)
            unique_successful_edges.append(edge)

    print(f"[INFO] Found {len(unique_successful_edges)} unique successful attacks.")

    # 3. Sample N edges
    sample_size = min(args.num_samples, len(unique_successful_edges))
    sampled_edges = random.sample(unique_successful_edges, sample_size)
    print(f"[INFO] Randomly sampled {sample_size} edges.")

    # 4. Prepare Embedder for automated similarity (if you want it in the Key file)
    print("[INFO] Warming up embedding cache for the Key file...")
    embedder = EmbeddingCache()
    all_texts = list({t for e in sampled_edges for t in (e.root_prompt, e.child_prompt)})
    embedder.batch_embed(all_texts)

    # 5. Build Blind and Key dictionaries
    blind_eval_data = []
    ground_truth_key = []

    for edge in sampled_edges:
        # Generate a unique 8-character ID to link the blind file to the key file
        eval_id = str(uuid.uuid4())[:8] 
        
        # Calculate Automated Similarity
        auto_sim = embedder.cosine(edge.root_prompt, edge.child_prompt)

        # File 1: What YOU will see and fill out
        blind_eval_data.append({
            "eval_id": eval_id,
            "root_prompt": edge.root_prompt,
            "mutated_prompt": edge.child_prompt,
            "depth": edge.refinement_iter,
            # --- FIELDS FOR YOU TO FILL IN BY HAND ---
            "human_is_harmful_intent_preserved": None, # e.g., 1 (Yes) or 0 (No)
            "human_similarity_score": None,            # e.g., 1 to 5 scale
            "notes": ""                                # Optional qualitative notes
        })

        # File 2: The hidden data for later comparison
        ground_truth_key.append({
            "eval_id": eval_id,
            "root_prompt": edge.root_prompt,
            "automated_success": edge.success,
            "automated_semantic_similarity": auto_sim,
            "depth": edge.refinement_iter,
            "operator_used": {edge.operator_name, edge.operator_type, edge.operator_metadata}
        })

    # 6. Save the files
    blind_path = os.path.join(args.out_dir, f"blind_evaluation_set_{sample_size}.json")
    key_path = os.path.join(args.out_dir, f"ground_truth_key_{sample_size}.json")

    with open(blind_path, "w") as f:
        json.dump(blind_eval_data, f, indent=4)
        
    with open(key_path, "w") as f:
        json.dump(ground_truth_key, f, indent=4)

    print(f"\n[SUCCESS] Extraction complete.")
    print(f"-> Blind Evaluation File: {blind_path}")
    print(f"-> Ground Truth Key File: {key_path}")
    print("\nIMPORTANT: Only open the 'blind_evaluation_set' file. Do not look at the key file until grading is complete!")

if __name__ == "__main__":
    main()