import hashlib, os, subprocess, sys, tempfile

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import gen_sudoku
from data_sudoku import SudokuBoards, load_test_boards
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from train import Trainer, _all_units_valid, build, load_config, state_hash

TINY = ["model.n_layer=1", "model.n_embd=64", "model.n_head=4", "train.precision=fp32", "train.warmup=2"]
CPU = dict(os.environ, CUDA_VISIBLE_DEVICES="")


def released(d="medium"):
    c = OmegaConf.load(f"config/eval/sudoku-{d}.yaml")
    c.tokenizer.vqvae.checkpoint_path = f"artifacts/sudoku/{d}/tokenizer.pt"
    v = VQVAETokenizer.from_config(c.tokenizer).vqvae.eval()
    n = NextConceptTransformer(vqvae_local=v, config=c.model)
    n.load_state_dict(torch.load(f"artifacts/sudoku/{d}/generator.pt", map_location="cpu", weights_only=True)["model"], strict=True)
    return v, n.eval()


def test_board_pool(pool):
    puzzle, solution = SudokuBoards(pool, ["easy", "medium"]).sample(64, np.random.default_rng(0))
    assert puzzle.shape == (64, 91) and solution.shape == (64, 89)
    assert (puzzle[:, 0] == 11).all() and (puzzle[:, 90] == 11).all()
    body = puzzle[:, 1:90]
    assert ((body == solution) | (body == 0)).all(), "givens must match the solution"
    keep = np.arange(89) % 10 != 9
    assert _all_units_valid(solution[:, keep]).all(), "augmented solutions must stay valid"
    assert set((body[:, keep] != 0).sum(1).tolist()) <= {40, 35}
    assert all(gen_sudoku.count_solutions(p[keep].tolist()) == 1 for p in body[:16]), "must stay unique"
    with np.load(pool) as z:
        assert (z["hard"] <= z["medium"]).all() and (z["medium"] <= z["easy"]).all(), "clue sets must nest"
        assert (z["hard"].sum(1) == 30).all() and (z["easy"].sum(1) == 40).all()
        grids = {tuple(g) for g in z["solutions"].tolist()}
    tests = {tuple(int(t) for t in s if t != 10) for s in load_test_boards("data/sudoku", "hard")[1]}
    assert not grids & tests, "pool overlaps the test grids"


def test_soft_path_and_rollout_match_released_code():
    v, n = released()
    puzzles, solutions = load_test_boards("data/sudoku", "medium", 8)
    x, s = torch.tensor(puzzles), torch.tensor(solutions)
    with torch.no_grad():
        ids = v.encode(s)
        assert torch.equal(v.vq.soft_idx_to_z_hat(F.one_hot(ids, 16).float()), v.vq.idx_to_z_q(ids))
        z_prefix = v.encoder.wte(x)
        full = n(torch.cat([z_prefix, n.build_l0_query_input(8), v.vq.idx_to_ncm_input(ids)], dim=1))
        levels = list(v.vq.levels)
        z_hat = 0
        for k in range(len(levels)):
            a = sum(levels[:k])
            if k:
                one = n.rollout_level_logits(v.vq.z_hat_to_ncm_input(z_hat, k), z_prefix, k)
                assert (one - full[:, a:a + levels[k]]).abs().max() < 1e-3, f"rollout mismatch at level {k}"
            z_hat = z_hat + v.vq.soft_level_to_z(F.one_hot(ids[:, a:a + levels[k]], 16).float(), k)


def test_e2e_grads_reach_generator_only(pool):
    for mode in ("teacher", "rollout"):
        cfg = load_config("config/train/pilot_E.yaml", TINY + [f"train.train_pool={pool}", f"train.e2e_mode={mode}"])
        v, n = build(cfg, torch.device("cpu"))
        before = state_hash(v)
        tr = Trainer(cfg, v, n, torch.device("cpu"))
        p, s = SudokuBoards(pool, ["easy"]).sample(4, np.random.default_rng(0))
        # Supervision off: any generator gradient can only come from the token loss.
        tr.keep = torch.zeros_like(tr.keep)
        loss, m = tr.loss(torch.tensor(p), torch.tensor(s), 0)
        loss.backward()
        assert m["mask_rate"] == 1.0 and m["supervised"] == 0.0
        assert all(q.grad is None for q in v.parameters()), "tokenizer received gradients"
        assert sum(q.grad.abs().sum().item() for q in n.parameters() if q.grad is not None) > 0, mode
        assert state_hash(v) == before


def test_baseline_loss_matches_eq10(pool):
    cfg = load_config("config/train/pilot_A.yaml", TINY + [f"train.train_pool={pool}", "train.corrupt_start=0", "train.corrupt_end=0"])
    v, n = build(cfg, torch.device("cpu"))
    n.eval()
    tr = Trainer(cfg, v, n, torch.device("cpu"))
    assert abs((tr.w * tr.lengths).sum().item() - 1.0) < 1e-5
    p, s = SudokuBoards(pool, ["easy"]).sample(4, np.random.default_rng(0))
    p, s = torch.tensor(p), torch.tensor(s)
    with torch.no_grad():
        loss, m = tr.loss(p, s, 0)
        ids = v.encode(s)
        logits = n(torch.cat([v.encoder.wte(p), n.build_l0_query_input(4), v.vq.idx_to_ncm_input(ids)], dim=1))
        want, off = 0.0, 0
        for lvl in v.vq.levels:
            ce = F.cross_entropy(logits[:, off:off + lvl].reshape(-1, 16), ids[:, off:off + lvl].reshape(-1))
            want += ce.item() / len(v.vq.levels)
            off += lvl
    assert abs(loss.item() - want) < 1e-4, (loss.item(), want)
    assert "token_ce" not in m and m["mask_rate"] == 0.0


def test_warm_start_and_per_level_mask(pool):
    cfg = load_config("config/train/pilot_I.yaml", TINY + [f"train.train_pool={pool}", "train.steps=100"])
    v, n = build(cfg, torch.device("cpu"))
    tr = Trainer(cfg, v, n, torch.device("cpu"))
    p, s = SudokuBoards(pool, ["easy"]).sample(4, np.random.default_rng(0))
    p, s = torch.tensor(p), torch.tensor(s)
    with torch.no_grad():
        _, m = tr.loss(p, s, 24)
        assert m["mask_rate"] == 0.0 and m["guidance"] == 1.0 and "token_ce" not in m
        _, m = tr.loss(p, s, 25)
        assert m["guidance"] == 1.0 and "token_ce" in m
        _, m = tr.loss(p, s, 99)
        assert m["mask_rate"] > 0.0 and m["guidance"] < 0.11
    cfg = load_config("config/train/pilot_H.yaml", TINY + [f"train.train_pool={pool}"])
    v, n = build(cfg, torch.device("cpu"))
    keep = Trainer(cfg, v, n, torch.device("cpu")).keep.tolist()
    assert keep == [1.0] * 6 + [0.5] * 6, keep


def _train(out, pool, run, steps=10):
    cmd = [sys.executable, "train.py", "--config", "config/train/smoke.yaml", *TINY, f"train.train_pool={pool}",
           f"train.output_root={out}", f"train.run_name={run}", f"train.steps={steps}", "train.batch_size=8",
           "train.ckpt_every=5", "train.eval_every=100000", "train.eval_boards=2", "train.final_eval_boards=2"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=CPU)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return r.stdout


def _final_hash(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


def test_resume_is_exact(pool, out):
    _train(out, pool, "straight")
    _train(out, pool, "cut")
    # Simulate a crash after step 5: drop everything newer, then resume.
    cut = os.path.join(out, "cut")
    assert sorted(f for f in os.listdir(cut) if f.startswith("ckpt_")) == ["ckpt_0000005.pt", "ckpt_0000010.pt"]
    os.remove(os.path.join(cut, "ckpt_0000010.pt"))
    os.remove(os.path.join(cut, "generator.pt"))
    log = _train(out, pool, "cut")
    assert "at step 5" in log, log[-500:]
    assert _final_hash(os.path.join(out, "straight", "generator.pt")) == _final_hash(os.path.join(cut, "generator.pt"))


def test_max_hours_exits_cleanly(pool, out):
    r = subprocess.run([sys.executable, "train.py", "--config", "config/train/smoke.yaml", "--max_hours", "0", *TINY,
                        f"train.train_pool={pool}", f"train.output_root={out}", "train.run_name=limit", "train.batch_size=4"],
                       capture_output=True, text=True, env=CPU)
    assert r.returncode == 0 and "stopping cleanly at step 1" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
    assert "ckpt_0000001.pt" in os.listdir(os.path.join(out, "limit"))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        pool = os.path.join(tmp, "pool.npz")
        subprocess.run([sys.executable, "scripts/gen_sudoku.py", "--out", pool, "--n", "64", "--workers", "4"], check=True)
        for name, fn in [
            ("board pool", lambda: test_board_pool(pool)),
            ("soft path and rollout", test_soft_path_and_rollout_match_released_code),
            ("e2e gradients", lambda: test_e2e_grads_reach_generator_only(pool)),
            ("baseline loss", lambda: test_baseline_loss_matches_eq10(pool)),
            ("warm start and per-level mask", lambda: test_warm_start_and_per_level_mask(pool)),
            ("exact resume", lambda: test_resume_is_exact(pool, tmp)),
            ("max_hours", lambda: test_max_hours_exits_cleanly(pool, tmp)),
        ]:
            fn()
            print(f"PASS {name}", flush=True)
