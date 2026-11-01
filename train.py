import argparse, glob, hashlib, json, math, os, random, signal, time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from data_sudoku import SudokuBoards, load_test_boards
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

# Every extension to the baseline recipe defaults to off.
DEFAULTS = {
    "run_name": "run",
    "seed": 42,
    "output_root": "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs",
    # data
    "tokenizer_dir": "artifacts/sudoku/medium",
    "test_data_dir": "data/sudoku",
    "train_pool": "data/sudoku_train.npz",
    "train_difficulties": ["easy", "medium"],
    "eval_difficulties": ["easy", "medium", "hard"],
    # recipe: paper Table 6, except alpha and corruption, which the paper does not give for Sudoku
    "steps": 10000,
    "batch_size": 256,
    "grad_accum": 1,
    "lr": 3.0e-4,
    "lr_floor": 3.0e-5,
    "warmup": 1000,
    "betas": [0.9, 0.99],
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "alpha": 1.0,
    "corrupt_start": 0.75,
    "corrupt_end": 0.25,
    # sparse level supervision: probability that a level's CE term is dropped (scalar or per-level list)
    "level_mask_prob": 0.0,
    "level_mask_schedule": "per_example",  # per_example | per_batch
    # end-to-end token loss through the frozen decoder
    "e2e_weight": 0.0,
    "e2e_estimator": "gumbel",  # gumbel | ste
    "gumbel_tau": 1.0,
    "e2e_mode": "teacher",  # teacher | rollout
    "e2e_rollout_levels": 3,
    # guidance weight lambda(t)
    "guidance_anneal": "none",  # none | linear | cosine
    "guidance_start": 1.0,
    "guidance_end": 0.1,
    "anneal_steps": 0,  # 0 means train.steps
    # warm start: for this fraction of steps, full supervision with no masking and no token loss
    "warm_start_frac": 0.0,
    # runtime
    "precision": "fp16",  # bf16 | fp16 | fp32
    "compile": False,
    "ckpt_every": 500,
    "keep_ckpts": 2,
    "eval_every": 0,  # 0 means steps // 4
    "eval_boards": 500,
    "final_eval_boards": 2000,
    "eval_batch_size": 128,
    "log_every": 20,
    "wandb": False,
    "wandb_project": "ncm-sparse-supervision",
}


def load_config(path, overrides):
    cfg = OmegaConf.load(path)
    base = OmegaConf.load(cfg.pop("base"))
    return OmegaConf.merge(base, {"train": DEFAULTS}, cfg, OmegaConf.from_dotlist(overrides))


def state_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build(cfg, device):
    t = cfg.train
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(t.tokenizer_dir, cfg.checkpoints.tokenizer)
    vqvae = VQVAETokenizer.from_config(cfg.tokenizer).vqvae.to(device).eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model).to(device)
    net.apply(net._init_weights)
    for pn, p in net.named_parameters():
        if pn.endswith("c_proj.weight"):
            torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.model.n_layer))
    return vqvae, net


def lr_at(step, t):
    warmup = min(t.warmup, t.steps // 10)
    if step < warmup:
        return t.lr * (step + 1) / warmup
    x = min((step - warmup) / max(t.steps - warmup, 1), 1.0)
    return t.lr_floor + 0.5 * (t.lr - t.lr_floor) * (1 + math.cos(math.pi * x))


def guidance_at(step, t):
    if t.guidance_anneal == "none":
        return float(t.guidance_start)
    warm = int(t.warm_start_frac * t.steps)
    x = min(max(step - warm, 0) / max((t.anneal_steps or t.steps) - warm, 1), 1.0)
    if t.guidance_anneal == "cosine":
        x = 0.5 * (1 - math.cos(math.pi * x))
    elif t.guidance_anneal != "linear":
        raise ValueError(f"unknown guidance_anneal {t.guidance_anneal}")
    return t.guidance_start + x * (t.guidance_end - t.guidance_start)


class Trainer:
    def __init__(self, cfg, vqvae, net, device):
        self.t, self.vqvae, self.net, self.device = cfg.train, vqvae, net, device
        self.vq = vqvae.vq
        self.levels = list(self.vq.levels)
        self.K = len(self.levels)
        self.offsets = [sum(self.levels[:k]) for k in range(self.K)]
        t = self.t
        lengths = torch.tensor(self.levels, dtype=torch.float32, device=device)
        # Eq. 10: w_k = l_k^-alpha / sum_j l_j^(1-alpha), a per-position weight.
        self.w = lengths.pow(-t.alpha) / lengths.pow(1 - t.alpha).sum()
        self.lengths = lengths
        # Level K's codes are never an input, so corruption spans levels 1..K-1.
        self.corrupt = np.linspace(t.corrupt_start, t.corrupt_end, self.K - 1).tolist()
        p = t.level_mask_prob
        p = [float(p)] * self.K if isinstance(p, (int, float)) else [float(x) for x in p]
        assert len(p) == self.K, f"level_mask_prob needs {self.K} entries"
        assert max(p) == 0 or t.e2e_weight > 0, "masked levels need e2e_weight > 0 for any signal"
        self.keep = 1.0 - torch.tensor(p, device=device)
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[t.precision]
        self.forward_net = torch.compile(net) if t.compile else net

    def autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None)

    def no_autocast(self):
        return torch.autocast(self.device.type, enabled=False)

    @torch.no_grad()
    def corrupt_codes(self, ids):
        out = ids.clone()
        for k, p in enumerate(self.corrupt):
            a, b = self.offsets[k], self.offsets[k] + self.levels[k]
            hit = torch.rand(ids.shape[0], self.levels[k], device=ids.device) < p
            noise = torch.randint(0, self.vq.vqvae_vocab_sizes[k], hit.shape, device=ids.device)
            out[:, a:b] = torch.where(hit, noise, ids[:, a:b])
        return out

    def sample_mask(self, B):
        shape = (B, self.K) if self.t.level_mask_schedule == "per_example" else (1, self.K)
        if self.t.level_mask_schedule not in ("per_example", "per_batch"):
            raise ValueError(f"unknown level_mask_schedule {self.t.level_mask_schedule}")
        return (torch.rand(shape, device=self.device) < self.keep).float().expand(B, -1)

    def codes_from_logits(self, logits):
        t = self.t
        if t.e2e_estimator == "gumbel":
            return F.gumbel_softmax(logits, tau=t.gumbel_tau, hard=True)
        if t.e2e_estimator != "ste":
            raise ValueError(f"unknown e2e_estimator {t.e2e_estimator}")
        soft = F.softmax(logits / t.gumbel_tau, dim=-1)
        return F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype) + soft - soft.detach()

    def token_loss(self, z_hat, solution):
        logits = self.vqvae.decoder(z_hat)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), solution.reshape(-1))

    def rollout_token_loss(self, ids, z_prefix, solution):
        V = self.net.max_vocab
        first = self.K - self.t.e2e_rollout_levels
        assert first >= 1, "e2e_rollout_levels must leave level 1 teacher-forced"
        with torch.no_grad():
            z_hat = 0
            for k in range(first):
                y = F.one_hot(ids[:, self.offsets[k]:self.offsets[k] + self.levels[k]], V).float()
                z_hat = z_hat + self.vq.soft_level_to_z(y, k)
        for k in range(first, self.K):
            x_level = self.vq.z_hat_to_ncm_input(z_hat, k)
            with self.autocast():
                logits = self.net.rollout_level_logits(x_level, z_prefix, k)
            z_hat = z_hat + self.vq.soft_level_to_z(self.codes_from_logits(logits.float()), k)
        return self.token_loss(z_hat, solution)

    def loss(self, puzzle, solution, step):
        t = self.t
        B = puzzle.shape[0]
        with self.no_autocast(), torch.no_grad():
            ids = self.vqvae.encode(solution)
            z_prefix = self.vqvae.encoder.wte(puzzle)
            targets = self.vq.idx_to_ncm_input(self.corrupt_codes(ids))
        # The level-0 queries are learned, so they join outside the no-grad block.
        x = torch.cat([z_prefix, self.net.build_l0_query_input(B, z_prefix.dtype), targets], dim=1)
        with self.autocast():
            logits = self.forward_net(x)
        with self.no_autocast():
            logits = logits.float()
            ce_pos = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction="none").view(B, -1)
            ce_lvl = torch.stack([ce_pos[:, a:a + l].sum(1) for a, l in zip(self.offsets, self.levels)], dim=1)
            warm = step < int(t.warm_start_frac * t.steps)
            mask = torch.ones_like(ce_lvl) if warm else self.sample_mask(B)
            guidance = guidance_at(step, t)
            supervised = (mask * self.w * ce_lvl).sum(1).mean()
            total = guidance * supervised
            tok = None
            if t.e2e_weight > 0 and not warm:
                if t.e2e_mode == "teacher":
                    tok = self.token_loss(self.vq.soft_idx_to_z_hat(self.codes_from_logits(logits)), solution)
                elif t.e2e_mode == "rollout":
                    tok = self.rollout_token_loss(ids, z_prefix, solution)
                else:
                    raise ValueError(f"unknown e2e_mode {t.e2e_mode}")
                total = total + t.e2e_weight * tok
        metrics = {"loss": total.item(), "supervised": supervised.item(), "guidance": guidance,
                   "mask_rate": 1.0 - mask.mean().item(),
                   "ce_per_level": (ce_lvl.mean(0) / self.lengths).tolist()}
        if tok is not None:
            metrics["token_ce"] = tok.item()
        return total, metrics


def _all_units_valid(grids):
    rows = grids.reshape(-1, 9, 9)
    cols = rows.transpose(0, 2, 1)
    boxes = rows.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    want = np.arange(1, 10)
    return np.all([(np.sort(u, axis=-1) == want).all(axis=(1, 2)) for u in (rows, cols, boxes)], axis=0)


@torch.no_grad()
def evaluate(net, vqvae, t, n, device):
    was_training = net.training
    net.eval()
    results = {}
    keep = np.arange(89) % 10 != 9
    for d in t.eval_difficulties:
        puzzles, solutions = load_test_boards(t.test_data_dir, d, n)
        preds = []
        for s in range(0, len(puzzles), t.eval_batch_size):
            x = torch.tensor(puzzles[s:s + t.eval_batch_size], device=device)
            codes = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
            preds.append(vqvae.decoder(vqvae.vq.idx_to_z_q(codes)).argmax(-1).cpu().numpy())
        preds = np.concatenate(preds)
        results[d] = {"solved": 100.0 * float((preds == solutions).all(1).mean()),
                      "cell": 100.0 * float((preds[:, keep] == solutions[:, keep]).mean()),
                      "valid": 100.0 * float(_all_units_valid(preds[:, keep]).mean()),
                      "boards": len(puzzles)}
    net.train(was_training)
    return results


def latest_checkpoint(path):
    if os.path.isfile(path):
        return path
    found = sorted(glob.glob(os.path.join(path, "ckpt_*.pt")))
    return found[-1] if found else None


def save_checkpoint(run_dir, step, net, opt, scaler, data_rng, wandb_id, keep):
    state = {"model": net.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
             "wandb_id": wandb_id, "rng": {"python": random.getstate(), "numpy": data_rng.bit_generator.state,
                                           "torch": torch.get_rng_state(),
                                           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    path = os.path.join(run_dir, f"ckpt_{step:07d}.pt")
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    for old in sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))[:-keep]:
        os.remove(old)
    return path


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser(description="train the NCM generator; overrides use dotted keys, e.g. train.steps=200")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume_from", default="", help="checkpoint file or directory; default: latest in the run dir")
    ap.add_argument("--max_hours", type=float, default=11.5, help="save and exit cleanly after this many hours")
    ap.add_argument("--benchmark", type=int, default=0, help="time this many updates, print seconds per update, exit")
    a, overrides = ap.parse_known_args()

    cfg = load_config(a.config, overrides)
    t = cfg.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(t.seed); np.random.seed(t.seed); torch.manual_seed(t.seed)
    data_rng = np.random.default_rng(t.seed)

    vqvae, net = build(cfg, device)
    tok_hash = state_hash(vqvae)
    trainer = Trainer(cfg, vqvae, net, device)
    data = SudokuBoards(t.train_pool, list(t.train_difficulties))
    decay = [p for p in net.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in net.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=t.lr, betas=tuple(t.betas))
    scaler = torch.amp.GradScaler(device.type, enabled=t.precision == "fp16" and device.type == "cuda")

    run_dir = os.path.join(t.output_root, t.run_name)
    step, wandb_id, resume = 0, None, None
    if not a.benchmark:
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
        # A checkpoint in this run's own dir is newer than any attached input.
        resume = latest_checkpoint(run_dir) or (latest_checkpoint(a.resume_from) if a.resume_from else None)
        if a.resume_from and not resume:
            raise SystemExit(f"no checkpoint found at {a.resume_from}")
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, wandb_id = state["step"], state["wandb_id"]
        random.setstate(state["rng"]["python"])
        data_rng.bit_generator.state = state["rng"]["numpy"]
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if state["rng"]["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
        print(f"[train] resumed {resume} at step {step}", flush=True)

    wandb_run = None
    if t.wandb and os.environ.get("WANDB_API_KEY") and not a.benchmark:
        import wandb
        wandb_id = wandb_id or wandb.util.generate_id()
        wandb_run = wandb.init(project=t.wandb_project, name=t.run_name, id=wandb_id, resume="allow",
                               config=OmegaConf.to_container(cfg, resolve=True))

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    eval_every = t.eval_every or max(t.steps // 4, 1)
    total_steps = a.benchmark or t.steps
    start, timer, timed = time.time(), None, 0
    net.train()
    print(f"[train] {t.run_name}: device={device} precision={t.precision} steps={total_steps} "
          f"batch={t.batch_size}x{t.grad_accum}", flush=True)

    while step < total_steps:
        tick = time.time()
        for group in opt.param_groups:
            group["lr"] = lr_at(step, t)
        opt.zero_grad(set_to_none=True)
        for _ in range(t.grad_accum):
            puzzle, solution = data.sample(t.batch_size, data_rng)
            puzzle = torch.as_tensor(puzzle, device=device)
            solution = torch.as_tensor(solution, device=device)
            loss, metrics = trainer.loss(puzzle, solution, step)
            if not math.isfinite(metrics["loss"]):
                raise SystemExit(f"non-finite loss at step {step}: {metrics}")
            scaler.scale(loss / t.grad_accum).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        step_time = time.time() - tick

        if a.benchmark:
            # The first updates include allocation and warm-up.
            if step == min(5, total_steps - 1):
                timer = time.time()
            elif timer is not None:
                timed += 1
            continue

        if step % t.log_every == 0 or step == 1:
            row = {"step": step, "lr": opt.param_groups[0]["lr"], "grad_norm": float(grad_norm), "step_time": step_time,
                   "elapsed_h": (time.time() - start) / 3600, **metrics}
            append_jsonl(os.path.join(run_dir, "metrics.jsonl"), row)
            print(f"step {step} loss {metrics['loss']:.4f} sup {metrics['supervised']:.4f} "
                  f"tok {metrics.get('token_ce', float('nan')):.4f} lambda {metrics['guidance']:.3f} "
                  f"mask {metrics['mask_rate']:.2f} grad {float(grad_norm):.2f} {step_time:.2f}s/step", flush=True)
            if wandb_run:
                flat = {k: v for k, v in row.items() if k != "ce_per_level"}
                flat.update({f"ce/level_{k + 1}": v for k, v in enumerate(metrics["ce_per_level"])})
                wandb_run.log(flat, step=step)

        hit_limit = stop["flag"] or (time.time() - start) / 3600 >= a.max_hours
        if step % t.ckpt_every == 0 or hit_limit or step == total_steps:
            save_checkpoint(run_dir, step, net, opt, scaler, data_rng, wandb_id, t.keep_ckpts)
            assert state_hash(vqvae) == tok_hash, "tokenizer weights changed"
        if hit_limit:
            print(f"[train] stopping cleanly at step {step}; resume to continue", flush=True)
            break
        if step % eval_every == 0 and step != total_steps:
            run_eval(net, vqvae, t, t.eval_boards, device, run_dir, step, wandb_run)

    if a.benchmark:
        spu = (time.time() - timer) / max(timed, 1)
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(json.dumps({"seconds_per_update": spu, "peak_gpu_gb": peak, "timed_updates": timed}), flush=True)
        return

    if step == total_steps:
        run_eval(net, vqvae, t, t.final_eval_boards, device, run_dir, step, wandb_run)
        torch.save({"model": net.state_dict()}, os.path.join(run_dir, "generator.pt"))
        print(f"[train] finished at step {step}", flush=True)
    if wandb_run:
        wandb_run.finish()


def run_eval(net, vqvae, t, n, device, run_dir, step, wandb_run):
    tick = time.time()
    results = evaluate(net, vqvae, t, n, device)
    for d, r in results.items():
        append_jsonl(os.path.join(run_dir, "eval.jsonl"), {"step": step, "difficulty": d, **r})
        print(f"[eval] step {step} {d}: solved {r['solved']:.2f} cell {r['cell']:.2f} valid {r['valid']:.2f} "
              f"({r['boards']} boards)", flush=True)
        if wandb_run:
            wandb_run.log({f"eval/{d}_solved": r["solved"], f"eval/{d}_cell": r["cell"], f"eval/{d}_valid": r["valid"]}, step=step)
    print(f"[eval] took {time.time() - tick:.0f}s", flush=True)


if __name__ == "__main__":
    main()
