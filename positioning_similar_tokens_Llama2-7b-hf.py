import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import warnings
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", type=str, default='cos_sim')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    warnings.filterwarnings(
        "ignore",
        message="`clean_up_tokenization_spaces` was not set.*",
        category=FutureWarning,
        module="transformers.tokenization_utils_base"
    )
    
    # Step 1: Load the pre-trained LLaMA2 7B model and tokenizer
    cache_dir = "/extdata1/donghwan/huggingface"
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir)
    
    # You can verify the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_size = len(tokenizer)
    all_token_ids = list(range(vocab_size))
    
    # Step 2: Arrange all embeddings so that tokens with similar embeddings are close to each other using a relaxed TSP
    # Obtain the embeddings for all tokens
    with torch.no_grad():
        # Access the embedding layer
        embedding_layer = model.get_input_embeddings()
        print(f'embedding_layer: {embedding_layer}')
        embeddings = embedding_layer.weight.data  # embeddings remain on device
        print(f'The shape of embeddings: {embeddings.shape}')
    
    # Check for NaN or inf values in embeddings
    if torch.any(torch.isnan(embeddings)) or torch.any(torch.isinf(embeddings)):
        print("Warning: embeddings contain NaN or Inf values!")
    else:
        print("No NaN and inf.")
    
    
    # Step 3: Implement Nearest Neighbor heuristic for TSP
    def nearest_neighbor_tsp(embeddings, tokenizer, norm):
        
        if args.norm == 'cos_sim':
            # Normalize embeddings for cosine similarity
            norms = embeddings.norm(p=2, dim=1, keepdim=True)
            epsilon = 1e-8
            norms = norms + epsilon  # Add epsilon to norms to avoid division by zero
            embeddings = embeddings / norms

        num_nodes = embeddings.shape[0]
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=embeddings.device)
        tour = []
        
        # Start from the first node (you can choose any starting point)
        current_idx = 0
        tour.append(current_idx)
        visited[current_idx] = True
        
        for step in range(1, num_nodes):
            # Compute cosine distances to all unvisited nodes
            current_embedding = embeddings[current_idx].unsqueeze(0)  # Shape [1, embedding_dim]
            
            if args.norm == 'cos_sim':
                distances = 1 - torch.matmul(embeddings, current_embedding.t()).squeeze()  # Shape [num_nodes]
            elif args.norm == 'l2':
                distances = torch.norm(embeddings - current_embedding, p=2, dim=1)
            else:
                raise ValueError("Invalid norm type.")
                
            distances[visited] = float('inf')  # Ignore visited nodes
            
            # Find the nearest unvisited neighbor
            next_idx = torch.argmin(distances).item()
            tour.append(next_idx)
            visited[next_idx] = True
            current_idx = next_idx
            
            # Optional: Print progress every 1000 steps
            if step % 1000 == 0:
                print(f"Visited {step} nodes...")
        
        return tour
        
    original_idx = [i for i in range(vocab_size)]
    original_tokens = tokenizer.convert_ids_to_tokens(original_idx)
    # Step 4: Save sorted tokens to a file
    with open(f"original_tokens_LLaMA2-7B-hf_{args.norm}.txt", "w", encoding="utf-8") as f:
        for token in original_tokens:
            f.write(f"{token}\n")
    print(f"\nOriginal token saved to 'sorted_tokens_LLaMA2_7B-hf_{args.norm}.txt'.")
    
    print(f"\nComputing token ordering using Nearest Neighbor heuristic with {args.norm}...")
    tour = nearest_neighbor_tsp(embeddings, tokenizer=tokenizer, norm=args.norm)
        
    tour_json = [int(i) for i in tour]
    with open(f'llama2-7b-hf_sorted_idx_{args.norm}.json', 'w') as f:
        json.dump(tour_json, f)

    # Output the sorted tokens
    sorted_tokens = tokenizer.convert_ids_to_tokens(tour)
    
    print(f"\nToken ordering saved to 'sorted_tokens_LLaMA2_7B-hf_{args.norm}.txt'.")
    
    # Optional: Save sorted tokens to a file
    with open(f"sorted_tokens_LLaMA2-7B-hf_{args.norm}.txt", "w", encoding="utf-8") as f:
        for token in sorted_tokens:
            f.write(f"{token}\n")
    
    print(f"\nToken ordering saved to 'sorted_tokens_LLaMA2_7B-hf_{args.norm}.txt'.")
    
if __name__ == "__main__":
    main()