#!/usr/bin/env python3
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Request:
    req_id: str
    prompt: str
    max_new_tokens: int
    arrival: float = field(default_factory=time.time)
    generated_ids: List[int] = field(default_factory=list)
    past_key_values: Optional[Tuple] = None
    prefills_done: bool = False

    def remaining(self) -> int:
        return self.max_new_tokens - len(self.generated_ids)

    def finished(self) -> bool:
        return self.remaining() <= 0


def aging_priority(req: Request, now: float, w_time: float, w_len: float) -> float:
    return w_time * (now - req.arrival) - w_len * req.remaining()


def init_model(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def run_one_step(req: Request, tokenizer, model, device: torch.device) -> None:
    with torch.no_grad():
        if not req.prefills_done:
            input_ids = tokenizer(req.prompt, return_tensors="pt").input_ids.to(device)
            outputs = model(input_ids, use_cache=True)
            req.past_key_values = outputs.past_key_values
            next_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            req.generated_ids.append(next_id)
            req.prefills_done = True
            return

        input_ids = torch.tensor([[req.generated_ids[-1]]], device=device)
        outputs = model(
            input_ids,
            use_cache=True,
            past_key_values=req.past_key_values,
        )
        req.past_key_values = outputs.past_key_values
        next_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        req.generated_ids.append(next_id)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimal Ascend inference loop with an aging scheduler."
    )
    parser.add_argument("--model", required=True, help="HF model path or name.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--w-time", type=float, default=1.0)
    parser.add_argument("--w-len", type=float, default=1.0)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    if not hasattr(torch_npu, "npu") or not torch_npu.npu.is_available():
        raise RuntimeError("torch_npu is not available on this machine.")

    torch_npu.npu.set_device(args.device_id)
    device = torch.device("npu")

    tokenizer, model = init_model(args.model, device)

    queue = [
        Request("r1", "Hello from Ascend:", args.max_new_tokens),
        Request("r2", "Tell me a joke:", args.max_new_tokens),
    ]

    while queue:
        now = time.time()
        queue.sort(
            key=lambda r: aging_priority(r, now, args.w_time, args.w_len),
            reverse=True,
        )
        req = queue.pop(0)

        run_one_step(req, tokenizer, model, device)

        if req.finished():
            text = tokenizer.decode(req.generated_ids, skip_special_tokens=True)
            print(f"{req.req_id}: {text}")
        else:
            queue.append(req)


if __name__ == "__main__":
    main()
