import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import random
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length
    )
    # Copy input_ids to labels so we can do causal LM
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

parser = argparse.ArgumentParser(description="Training script for a machine learning model.")
parser.add_argument("--max_length", type=int, default=4096)
parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
parser.add_argument('--emb_forcing', action="store_true")
parser.add_argument('--lambda_', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=100)
parser.add_argument('--save_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument("--norm", default='cos_sim', type=str)
parser.add_argument("--seed", type=int, default="YOUR_SEED")
parser.add_argument("--dataset", type=str, default="YOUR_DATASET")
args = parser.parse_args()

# Set the seed
set_seed(seed=args.seed)

# Custom Trainer to override compute_loss
class CustomTrainer(Trainer):
    def __init__(self, *args, custom_param=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_args = custom_param
        self.emb_losses = []
        self.original_losses = []
        self.steps = 0
        self.new_order = self.extra_args.new_order
        self.temp_original_losses = []
        self.temp_emb_losses = []
        self.graph_losses = []

    # Originally used for other purpose, but not used in the paper.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        
        loss_kwargs = {}
        loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
        
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        
        return (loss, outputs) if return_outputs else loss

# Model/Data Paths
model_name = "meta-llama/Llama-2-7b-hf"
cache_dir = "YOUR_CACHE_DIR"

auth_token = "YOUR_HUGGINGFACE_TOKEN"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=auth_token)

# Add pad token if missing
tokenizer.pad_token = tokenizer.eos_token

# Load the custom sorted index list
with open(f'llama2-7b-hf_sorted_idx_{args.norm}.json', 'r') as f:
    sorted_idx = json.load(f)
old_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(sorted_idx)}
new_id_to_old_id = {new_id: old_id for new_id, old_id in enumerate(sorted_idx)}
new_order = [new_id_to_old_id[i] for i in range(len(sorted_idx))]
old_order = [old_id_to_new_id[i] for i in range(len(sorted_idx))]
args.new_order = new_order

# Load Base Model *on single GPU*
model = LlamaForCausalLM.from_pretrained(model_name,
                                        # device_map="auto",
                                        # torch_dtype=torch.float16,
                                        cache_dir=cache_dir,
                                        # force_download=True,
                                        token=auth_token
                                        )
model = model.to("cuda")

embedding_weights = model.model.embed_tokens.weight
print(f'Embedding weights: {embedding_weights.shape}')
emb_diff_terms = embedding_weights[:-1].shape[0] - 1

if args.norm == 'l2':
    emb_diff = embedding_weights[:-1] - embedding_weights[1:]
    emb_loss = ((emb_diff ** 2).sum() / emb_diff_terms).item()
elif args.norm == 'cos_sim':
    norm_weights = embedding_weights / embedding_weights.norm(p=2, dim=1, keepdim=True)
    cos_sims = (norm_weights[:-1] * norm_weights[1:]).sum(dim=1)  # Cosine similarities for each pair
    emb_loss = (1 - cos_sims.mean()).item()  # 1 - mean(cosine similarity)
print(f'Before IC emb_{args.norm}_loss: {emb_loss}')

new_embedding_weights = embedding_weights[new_order].clone()
if args.norm == 'l2':
    new_emb_diff = new_embedding_weights[:-1] - new_embedding_weights[1:]
    new_emb_loss = ((new_emb_diff ** 2).sum() / emb_diff_terms).item()
elif args.norm == 'cos_sim':
    new_norm_weights = new_embedding_weights / new_embedding_weights.norm(p=2, dim=1, keepdim=True)
    new_cos_sims = (new_norm_weights[:-1] * new_norm_weights[1:]).sum(dim=1)
    new_emb_loss = (1 - new_cos_sims.mean()).item()
print(f'After IC emb_{args.norm}_loss: {new_emb_loss}')

# LoRA config
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=2,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# [Reproducibility Note]
# This line is enabled to match the exact experimental settings reported in the paper.
# Training with 'requires_grad=True' ensures the optimization trajectory 
# follows the same path as our experiments.
model.base_model.model.model.embed_tokens.weight.requires_grad = True

model.print_trainable_parameters()
# Load Dataset
dataset = load_dataset("YOUR_DATASET", args.dataset, cache_dir=cache_dir)
dataset["train"] = dataset["train"].shard(num_shards=5, index=0)
dataset["validation"] = dataset["validation"].shard(num_shards=10, index=0)

# Preprocess (tokenize) data
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

args.output_dir = f"./results/fine-tuning/YOUR_PATH"

# TrainingArguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    eval_strategy="steps",
    eval_steps=50,
    max_steps=args.max_steps,
    save_steps=args.save_steps,
    logging_steps=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.lr,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=False,
    load_best_model_at_end=False,
    report_to="none",
    dataloader_num_workers=8
)

# Create Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    custom_param=args
)

# Train
trainer.train()