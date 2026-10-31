import os

BASE = {"steps": 10000, "batch_size": 256}
SPARSE = {"level_mask_prob": 0.5}
E2E = {"e2e_weight": 1.0}
ANNEAL = {"guidance_anneal": "cosine", "guidance_start": 1.0, "guidance_end": 0.1}
ROLLOUT = {"e2e_mode": "rollout", "e2e_rollout_levels": 3}
RUNS = {
    "A": {},
    "B": {**SPARSE, **E2E, **ANNEAL},
    "C": {**SPARSE, **E2E},
    "D": {**E2E},
    "E": {**SPARSE, **E2E, **ANNEAL, **ROLLOUT},
}
SMOKE = {"steps": 200, "batch_size": 64, "precision": "fp32", "warmup": 20, "ckpt_every": 50, "log_every": 5,
         "eval_every": 100, "eval_boards": 32, "final_eval_boards": 64, "eval_batch_size": 32, **RUNS["E"]}


def write(name, fields):
    lines = ["base: config/eval/sudoku-medium.yaml", "train:", f"  run_name: {name}"]
    lines += [f"  {k}: {v}" for k, v in fields.items()]
    with open(os.path.join("config", "train", f"{name}.yaml"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


os.makedirs(os.path.join("config", "train"), exist_ok=True)
for run, extra in RUNS.items():
    write(f"pilot_{run}", {**BASE, **extra})
    write(f"full_{run}", {**BASE, "grad_accum": 4, **extra})
write("smoke", SMOKE)
