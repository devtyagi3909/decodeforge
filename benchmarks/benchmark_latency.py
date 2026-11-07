"""Microbenchmark suite for compilation, mask generation, and cached lookup latency."""

from __future__ import annotations

import time

import torch
from pydantic import BaseModel

from decodeforge.engine import GrammarEngine
from decodeforge.mask import apply_mask


class SamplePayload(BaseModel):
    user_id: int
    command: str
    is_admin: bool


def benchmark_suite(
    vocab_size: int = 32000, num_lookups: int = 20000
) -> dict[str, float]:
    # Synthetic vocabulary
    vocab = {i: f"token_{i}" for i in range(vocab_size)}
    vocab[0] = '{"user_id":'
    vocab[1] = "123"
    vocab[2] = ',"command":'
    vocab[3] = '"deploy"'
    vocab[4] = ',"is_admin":'
    vocab[5] = "true"
    vocab[6] = "}"
    vocab[7] = "<eos>"

    # 1. Compilation latency
    t0 = time.perf_counter()
    engine = GrammarEngine.from_pydantic(SamplePayload, vocab=vocab, eos_token_id=7)
    compilation_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Uncached first mask evaluation
    t0 = time.perf_counter()
    mask = engine.get_token_mask()
    first_mask_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Cached mask lookup latency (steady-state inference loop)
    t0 = time.perf_counter()
    for _ in range(num_lookups):
        _ = engine.get_token_mask()
    lookup_us = ((time.perf_counter() - t0) / num_lookups) * 1_000_000.0

    # 4. Mask application kernel latency
    logits = torch.randn(vocab_size)
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = apply_mask(logits, mask)
    mask_kernel_us = ((time.perf_counter() - t0) / 1000.0) * 1_000_000.0

    return {
        "vocab_size": float(vocab_size),
        "compilation_ms": compilation_ms,
        "first_mask_ms": first_mask_ms,
        "cached_lookup_us": lookup_us,
        "mask_kernel_us": mask_kernel_us,
    }


if __name__ == "__main__":
    print("Running DecodeForge benchmark...")
    results = benchmark_suite(vocab_size=32000)
    print(f"Vocab size: {int(results['vocab_size'])}")
    print(f"Compilation: {results['compilation_ms']:.2f} ms")
    print(f"First mask generation: {results['first_mask_ms']:.2f} ms")
    print(f"Cached lookup: {results['cached_lookup_us']:.3f} μs")
    print(f"Logits mask kernel: {results['mask_kernel_us']:.3f} μs")
