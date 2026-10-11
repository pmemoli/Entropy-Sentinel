import glob
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import RUNS_2048

CACHE = os.path.join(os.path.dirname(__file__), "cache")
MAX_LEN = 4096


@torch.no_grad()
def full_vocab_entropy(item, model, tokenizer, device):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}], add_generation_prompt=True, return_tensors="pt")[0].tolist()
    ids = (prompt + list(item["sequences"]))[:MAX_LEN]
    logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0]
    logp = F.log_softmax(logits[len(prompt) - 1:len(ids) - 1].float(), dim=-1)
    return (-(logp.exp() * logp).sum(-1)).cpu().numpy()


def main(bench, device="cuda:0"):
    os.makedirs(CACHE, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.bfloat16).to(device).eval()
    items = [it for f in sorted(glob.glob(f"{RUNS_2048}/phi3-3b-{bench}-test/*.pt")) for it in torch.load(f, map_location="cpu", weights_only=False)]
    profiles = [full_vocab_entropy(it, model, tokenizer, device) for it in items]
    torch.save({"profiles": profiles, "labels": [0 if it["success"] else 1 for it in items]}, f"{CACHE}/phi3-3b-{bench}-test.pt")


if __name__ == "__main__":
    main(*sys.argv[1:])
