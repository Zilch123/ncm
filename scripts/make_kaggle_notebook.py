import json, os, sys

CELLS = [
("markdown", """# NCM sparse level supervision: Kaggle run
Use **Save & Run All** with a T4 x2 accelerator and internet on. Set `MODE` and `RUNS` below.
Everything persistent lands in `/kaggle/working`. To resume, attach an earlier run's output as an input and set `RESUME_INPUT`."""),
("code", """MODE = "pilot"          # pilot | smoke
RUNS = ["pilot_A", "pilot_B"]   # first session A vs B; next pilot_C, pilot_D
SEED = 42
BUDGET_HOURS = 3.0      # the pilot step count is fitted to this
FIXED_STEPS = 0         # set to reuse an earlier session's step count (see /kaggle/working/fitted_steps.json)
RESUME_INPUT = ""       # e.g. "/kaggle/input/ncm-pilot-output"
REPO = "https://github.com/Zilch123/ncm"
BRANCH = "sparse-level-supervision"
"""),
("markdown", "## GPU check"),
("code", """import json, os, shutil, subprocess, sys, time

def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")
    return r.returncode

sh("nvidia-smi")
import torch
print("torch", torch.__version__, "gpus", torch.cuda.device_count())
assert torch.cuda.is_available(), "no usable GPU: choose T4 x2 (recent torch may not support the P100)"
# Exercises the GPU itself: an unsupported card fails here rather than hours later.
with torch.autocast("cuda", dtype=torch.float16):
    print("fp16 ok:", (torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")).dtype)
GPUS = torch.cuda.device_count()
"""),
("markdown", "## Clone, install, download"),
("code", """ROOT, ASSETS, OUT = "/tmp/ncm", "/tmp/assets", "/kaggle/working"
shutil.rmtree(ROOT, ignore_errors=True)
sh(f"git clone --depth 1 --branch {BRANCH} {REPO} {ROOT}")
os.chdir(ROOT)
sh("git log -1 --oneline")
sh("pip install -q omegaconf hydra-core 'huggingface_hub[cli]' wandb")
"""),
("code", """sh(f"hf download hazyresearch/ncm-checkpoints --include 'sudoku/*' --local-dir {ASSETS}/artifacts --quiet")
sh(f"hf download hazyresearch/ncm-datasets --repo-type dataset --include 'sudoku/*' --local-dir {ASSETS}/data --quiet")
POOL = f"{OUT}/data/sudoku_train.npz"
if not os.path.exists(POOL):
    sh(f"python scripts/gen_sudoku.py --out {POOL} --n 20000 --test-data-dir {ASSETS}/data/sudoku")
PATHS = (f"train.tokenizer_dir={ASSETS}/artifacts/sudoku/medium train.test_data_dir={ASSETS}/data/sudoku "
         f"train.train_pool={POOL} train.output_root={OUT} train.seed={SEED}")
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    PATHS += " train.wandb=true"
    print("W&B on")
except Exception:
    print("W&B off (no WANDB_API_KEY secret)")
os.makedirs(f"{OUT}/logs", exist_ok=True)
"""),
("markdown", "## T4 smoke (MODE = smoke)\nAll new features on, fp16, 200 steps, one hard kill and resume, then speed and memory at batch 256 for each kind of run."),
("code", """if MODE == "smoke":
    run = f"python train.py --config config/train/smoke.yaml {PATHS} train.precision=fp16 train.run_name=smoke_t4"
    log = f"{OUT}/logs/smoke_t4.log"
    p = subprocess.Popen(f"exec {run} >> {log} 2>&1", shell=True)
    while not os.path.exists(f"{OUT}/smoke_t4/ckpt_0000100.pt") and p.poll() is None:
        time.sleep(2)
    time.sleep(5)
    p.kill(); p.wait()
    print("killed; ckpts:", sorted(f for f in os.listdir(f"{OUT}/smoke_t4") if f.startswith("ckpt")))
    sh(f"{run} >> {log} 2>&1")
    sh(f"grep -E 'resumed|finished' {log}")
    sh(f"tail -n 25 {log}")
    for name in ("pilot_A", "pilot_B", "pilot_E"):
        sh(f"CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/{name}.yaml {PATHS} --benchmark 30 2>&1 | tail -n 1 | tee {OUT}/logs/bench_{name}.json")
"""),
("markdown", "## Fit the step count\nThe slowest run in the pair sets one shared count, so runs are compared at equal steps."),
("code", """STEPS = FIXED_STEPS
FITTED = f"{OUT}/fitted_steps.json"
if MODE == "pilot" and not STEPS:
    if RESUME_INPUT and os.path.exists(f"{RESUME_INPUT}/fitted_steps.json"):
        STEPS = json.load(open(f"{RESUME_INPUT}/fitted_steps.json"))["steps"]
    else:
        secs = []
        for name in RUNS:
            out = subprocess.run(f"CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/{name}.yaml {PATHS} --benchmark 30",
                                 shell=True, capture_output=True, text=True)
            if out.returncode != 0:
                raise RuntimeError(out.stdout[-2000:] + out.stderr[-2000:])
            res = json.loads(out.stdout.strip().splitlines()[-1])
            print(name, res, flush=True)
            secs.append(res["seconds_per_update"])
        # 15% is held back for evals and checkpoints; one GPU has to fit both runs in the budget.
        share = 1 if GPUS >= 2 else 2
        STEPS = max(500, min(10000, int(BUDGET_HOURS * 3600 * 0.85 / share / max(secs)) // 100 * 100))
    json.dump({"steps": STEPS, "gpus": GPUS}, open(FITTED, "w"))
print("steps:", STEPS)
"""),
("markdown", "## Launch\nOne run per GPU; progress is in `/kaggle/working/logs/<run>.log`."),
("code", """STATUS = 0
if MODE == "pilot":
    env = f"OUT={OUT} " + (f"RESUME_ROOT={RESUME_INPUT} " if RESUME_INPUT else "")
    extra = f"{PATHS} train.steps={STEPS}"
    STATUS = sh(f"{env} bash scripts/kaggle_launch.sh {RUNS[0]} {RUNS[1]} {extra}", check=False)
    for name in RUNS:
        sh(f"tail -n 15 {OUT}/logs/{name}.log", check=False)
"""),
("markdown", "## Results"),
("code", """extra = f"--extra-roots {RESUME_INPUT}" if RESUME_INPUT else ""
sh(f"python scripts/collect_results.py --root {OUT} {extra}", check=False)
shutil.rmtree(f"{OUT}/.cache", ignore_errors=True)
assert STATUS == 0, "a run failed; see the logs"
"""),
]


def build(mode="pilot", runs=None, steps=0):
    cells = []
    for kind, src in CELLS:
        if src.startswith('MODE = "pilot"'):
            src = src.replace('MODE = "pilot"', f'MODE = "{mode}"').replace("FIXED_STEPS = 0", f"FIXED_STEPS = {steps}")
            if runs:
                src = src.replace('["pilot_A", "pilot_B"]', json.dumps(runs))
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("notebooks", "kaggle_run.ipynb")
    mode = sys.argv[2] if len(sys.argv) > 2 else "pilot"
    runs = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        json.dump(build(mode, runs, steps), f, indent=1)
