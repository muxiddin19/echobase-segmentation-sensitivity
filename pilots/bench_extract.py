"""Choose the extraction model for TITAN Xp: fp16 vs fp32 label agreement and throughput on gold segments.

    python -m pilots.bench_extract --gpu 1 --n 256
"""
import argparse
import time

import torch

from .ami import gold_segments, meetings
from .build_versions import normalize
from .extract import Extractor

p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, default=1)
p.add_argument("--n", type=int, default=256)
a = p.parse_args()

texts = [normalize(s.text) for m in meetings("dev")[:3] for s in gold_segments(m) if s.text.strip()][:a.n]
results = {}
for model_id, dtype in [("Qwen/Qwen2.5-1.5B-Instruct", torch.float32), ("Qwen/Qwen2.5-1.5B-Instruct", torch.float16),
                        ("Qwen/Qwen2.5-3B-Instruct", torch.float16)]:
    ex = Extractor(model_id, gpu=a.gpu, dtype=dtype)
    torch.cuda.synchronize(a.gpu)
    t0 = time.time()
    out = ex("action_item", texts)
    torch.cuda.synchronize(a.gpu)
    dt = time.time() - t0
    results[ex.id] = out
    yes = sum(l == "yes" for l, _ in out)
    print(f"{ex.id}: {len(texts) / dt:.1f} segments/s, yes={yes}/{len(texts)}, "
          f"peak_mem={torch.cuda.max_memory_allocated(a.gpu) / 2**30:.1f} GiB", flush=True)
    del ex
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(a.gpu)

ids = list(results)
agree = sum(x[0] == y[0] for x, y in zip(results[ids[0]], results[ids[1]])) / len(texts)
print(f"1.5B fp32 vs fp16 label agreement: {agree:.3f}")
