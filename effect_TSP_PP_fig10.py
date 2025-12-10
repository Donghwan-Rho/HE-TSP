import os
import argparse
import numpy as np
import torch
import random
import json

from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.sampling_utils import amplifier, trigger_torch, post_processing

parser = argparse.ArgumentParser(description="Effect of TSP and PP.")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--cache_dir", default='/extdata1/donghwan/huggingface', type=str)
parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument("--ckpt", default=0, type=int)
parser.add_argument("--prompt", default='My name is ', type=str)
parser.add_argument("--samples", default=10000, type=int, help="number of random samples per checkpoint")
parser.add_argument("--tsp_index_json", default="llama2-7b-hf_sorted_idx_cos_sim.json", type=str,
                    help="Permutation json (new_id -> old_id). If missing, TSP metrics are skipped.")
args = parser.parse_args()

os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_name = args.base_model
cache_dir = args.cache_dir
CKPT = args.ckpt

@torch.no_grad()
def random_sampling_torch(p: torch.Tensor):
    # p = np.array(p, dtype=np.float64)
    r = torch.rand((), dtype=torch.float64, device=device)
    cumul_p = torch.cumsum(p, dim=0)
    cumul_p_minus_r = cumul_p - r
    amplified = amplifier(amplifier(amplifier(cumul_p_minus_r)))
    prev_amplified = torch.roll(amplified, 1)
    I = trigger_torch(prev_amplified, amplified)
    Iprime = post_processing(I)
    
    k = int(torch.searchsorted(cumul_p, r, right=False).item())
    if k == p.numel():
        k = p.numel() - 1
    return I, Iprime, k

def load_model_for_step():
    model = LlamaForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return model

print(f"[info] Loading tokenizer: {base_model_name}")
tokenizer = LlamaTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)

def get_embedding_weight(the_model) -> torch.Tensor:
    # robust path for base & PEFT
    if hasattr(the_model, "get_input_embeddings"):
        emb = the_model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight
    # fallbacks
    candidates = [
        "model.embed_tokens.weight",
        "base_model.model.model.embed_tokens.weight",
        "base_model.model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
    ]
    for path in candidates:
        try:
            obj = the_model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except Exception:
            continue
    raise AttributeError("Embedding weight not found.")

def softmax64(logits: torch.Tensor) -> torch.Tensor:
    m = logits.max()
    z = torch.exp(logits - m)
    return z / z.sum()

@torch.no_grad()
def probs_for_prompt(the_model, prompt: str, perm: torch.Tensor | None = None):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # logits를 바로 float64로 캐스팅
        logits = the_model(**inputs).logits[:, -1, :].squeeze().to(torch.float64)  # (V,)
    probs = softmax64(logits)  # float64 softmax
    if perm is not None:
        probs = probs.index_select(0, perm)
    return probs  # float64

def load_perm_tensor(model, tsp_index_json: str):
    if tsp_index_json and os.path.isfile(tsp_index_json):
        with open(tsp_index_json, "r") as f:
            sorted_idx = json.load(f)  # list: new_index -> old_id
        V_expected = getattr(model.config, "vocab_size", None)
        if V_expected is not None and len(sorted_idx) != V_expected:
            print(f"Perm length {len(sorted_idx)} != vocab_size {V_expected}. Proceed anyway.")
        return torch.tensor(sorted_idx, dtype=torch.long, device=device)
    else:
        print(f"TSP json not found: {tsp_index_json}. TSP metrics will be skipped.")
        return None

@torch.no_grad()
def run_experiment(mdl, probs: torch.Tensor, W: torch.Tensor, samples: int):
    V, D = W.shape

    s_errI  = torch.zeros((), device=device, dtype=torch.float64)
    s_errIp = torch.zeros((), device=device, dtype=torch.float64)
    s2_errI  = torch.zeros((), device=device, dtype=torch.float64)
    s2_errIp = torch.zeros((), device=device, dtype=torch.float64)

    s_cos_b = torch.zeros((), device=device, dtype=torch.float64)
    s_cos_a = torch.zeros((), device=device, dtype=torch.float64)
    s2_cos_b = torch.zeros((), device=device, dtype=torch.float64)
    s2_cos_a = torch.zeros((), device=device, dtype=torch.float64)

    zeroD = torch.zeros(D, device=device, dtype=torch.float64)

    for _ in range(samples):
        I, Iprime, k = random_sampling_torch(probs)

        temp = I.clone()
        temp[k] = temp[k] - 1.0
        errI = torch.max(torch.abs(temp))

        temp2 = Iprime.clone()
        temp2[k] = temp2[k] - 1.0
        errIp = torch.max(torch.abs(temp2))

        s_errI  += errI
        s_errIp += errIp
        s2_errI  += errI * errI
        s2_errIp += errIp * errIp

        nz_b = torch.nonzero(I, as_tuple=False).flatten()
        if nz_b.numel() == 0:
            v_hat_b = zeroD
        else:
            rows_b = W.index_select(0, nz_b)
            vals_b = I.index_select(0, nz_b).unsqueeze(1)
            v_hat_b = (vals_b * rows_b).sum(dim=0)

        nz_a = torch.nonzero(Iprime, as_tuple=False).flatten()
        if nz_a.numel() == 0:
            v_hat_a = zeroD
        else:
            rows_a = W.index_select(0, nz_a)
            vals_a = Iprime.index_select(0, nz_a).unsqueeze(1)
            v_hat_a = (vals_a * rows_a).sum(dim=0)

        v_true = W[k]

        denom_b = (torch.linalg.norm(v_hat_b) * torch.linalg.norm(v_true)).clamp_min(1e-18)
        denom_a = (torch.linalg.norm(v_hat_a) * torch.linalg.norm(v_true)).clamp_min(1e-18)
        cos_b = torch.dot(v_hat_b, v_true) / denom_b
        cos_a = torch.dot(v_hat_a, v_true) / denom_a

        one_minus_cos_b = 1.0 - cos_b
        one_minus_cos_a = 1.0 - cos_a

        s_cos_b  += one_minus_cos_b
        s_cos_a  += one_minus_cos_a
        s2_cos_b += one_minus_cos_b * one_minus_cos_b
        s2_cos_a += one_minus_cos_a * one_minus_cos_a

    n = torch.tensor(float(samples), device=device, dtype=torch.float64)

    mean_errI  = (s_errI  / n).item()
    mean_errIp = (s_errIp / n).item()
    mean_cos_b = (s_cos_b / n).item()
    mean_cos_a = (s_cos_a / n).item()

    var_errI  = torch.clamp((s2_errI  / n) - (s_errI  / n)**2, min=0.0).item()
    var_errIp = torch.clamp((s2_errIp / n) - (s_errIp / n)**2, min=0.0).item()
    var_cos_b = torch.clamp((s2_cos_b / n) - (s_cos_b / n)**2, min=0.0).item()
    var_cos_a = torch.clamp((s2_cos_a / n) - (s_cos_a / n)**2, min=0.0).item()

    std_errI  = float(np.sqrt(var_errI))
    std_errIp = float(np.sqrt(var_errIp))
    std_cos_b = float(np.sqrt(var_cos_b))
    std_cos_a = float(np.sqrt(var_cos_a))

    return (mean_errI, mean_errIp, mean_cos_b, mean_cos_a,
            std_errI, std_errIp, std_cos_b, std_cos_a)

def main():
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Ckpts: {CKPT}")
    print(f"Samples per ckpt: {args.samples}")

    model = load_model_for_step()

    # Without TSP
    probs_base = probs_for_prompt(model, args.prompt, perm=None).to(device=device, dtype=torch.float64)
    W_base = get_embedding_weight(model).detach().to(device=device, dtype=torch.float64)

    (errI_b, errIp_b, cos_b_b, cos_a_b,
        std_errI_b, std_errIp_b, std_cos_b_b, std_cos_a_b) = run_experiment(model, probs_base, W_base, args.samples)

    print(f"\nWithout TSP\n")
    print(f"Mean errors over {args.samples} samples")
    print(f"Without PP max-err(I) : {errI_b:.6e}")
    print(f"   With PP max-err(I'): {errIp_b:.6e}")
    print(f"Cosine distance mean")
    print(f"Without PP: {cos_b_b:.6e}")
    print(f"   With PP: {cos_a_b:.6e}")

    # With TSP (if available)
    perm = load_perm_tensor(model, args.tsp_index_json)
    if perm is not None:
        probs_tsp = probs_base.index_select(0, perm)
        W_tsp = W_base.index_select(0, perm)

        (errI_t, errIp_t, cos_b_t, cos_a_t,
            std_errI_t, std_errIp_t, std_cos_b_t, std_cos_a_t) = run_experiment(model, probs_tsp, W_tsp, args.samples)

        print(f"\nWith TSP\n")
        print(f"Mean errors over {args.samples} samples")
        print(f"Without PP max-err(I) : {errI_t:.6e}")
        print(f"   With PP max-err(I'): {errIp_t:.6e}")
        print(f"Cosine distance mean")
        print(f"Without PP: {cos_b_t:.6e}")
        print(f"   With PP: {cos_a_t:.6e}")
    else:
        print("TSP json not found — TSP metrics skipped.")

    # free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
