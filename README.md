# Encryption-friendly LLM Architecture
This is the repository of our paper [Traveling Salesman-Based Token Ordering Improves Stability in Homomorphically Encrypted Language Models](https://arxiv.org/pdf/2510.12343).

First authors: **Donghwan Rho** (Seoul National University) and **Sieun Seo** (Ewha Womans University) 

Our experiments were run using the `Llama2-7b-hf` model. You can our codes with other models with appropriate modifications.

## 1. Token ordering with TSP

To reorder tokens with TSP, run

```
python positioning_similar_tokens_Llama2-7b-hf.py
```

After you run this code, the following files are created.
* llama2-7b-hf_sorted_idx_cos_sim.json
* original_tokens_LLaMA2-7B-hf_cos_sim.txt
* sorted_tokens_LLaMA2-7B-hf_cos_sim.txt

Comparing two `.txt` files, you can see the original and reordered tokens.

## 2. Fine-tuning (Optional)

If you want to generate text with a fine-tuned model, then run `llama_finetune_singleGPU.py`

For reproduction, run

```
python llama2_finetune_singleGPU.py \
    --cache_dir huggingface \
    --max_length 4096 \
    --gradient_accumulation_steps 32 \
    --max_steps 100 \
    --lr 5e-5 \
    --norm cos_sim \
    --seed 42 \
    --dataset wikitext-103-v1
```

## 3. Text generation with TSP and post-processing

To generate text with the sampling algorithm for CKKS, a homomorphic encryption scheme, run `text_generation.py`

For reproduction, for example, run

```
python text_generation.py \
    --sampling weighted \
    --finetuned \
    --model_dir wikitext-103-v1/maxlength4096_lr5e-05_grad-acc32_steps100_lambda1_cos_sim_seed \
    --cache_dir huggingface \
    --hf_token XXX \
    --train_step 60 \
    --post_processed \
    --index_changed \
    --max_length 1500 \
    --num_generation 100 \
    --prompt "Please introduce yourself." \
    --norm cos_sim \
    --seed 42 \
    --text_seed 42
```

To compute the corruption score ratio in the paper, you need a `GPT-4` API in `utils/gpt4_corruption_score.py`.