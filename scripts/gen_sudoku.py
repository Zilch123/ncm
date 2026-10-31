import argparse, multiprocessing, os, pickle, random, sys, time

import numpy as np

CLUES = {"easy": 40, "medium": 35, "hard": 30}


def _peers():
    box = lambda r, c: r // 3 * 3 + c // 3
    return [(i // 9, i % 9, box(i // 9, i % 9)) for i in range(81)]


PEERS = _peers()


def fill(rng):
    grid, rows, cols, boxes = [0] * 81, [0] * 9, [0] * 9, [0] * 9

    def rec(rem):
        if not rem:
            return True
        best, best_n = 0, 10
        for j, i in enumerate(rem):
            r, c, b = PEERS[i]
            n = 9 - bin(rows[r] | cols[c] | boxes[b]).count("1")
            if n < best_n:
                best, best_n = j, n
        i = rem[best]
        r, c, b = PEERS[i]
        digits = list(range(1, 10))
        rng.shuffle(digits)
        for v in digits:
            m = 1 << v
            if (rows[r] | cols[c] | boxes[b]) & m:
                continue
            rows[r] |= m; cols[c] |= m; boxes[b] |= m; grid[i] = v
            if rec(rem[:best] + rem[best + 1:]):
                return True
            rows[r] ^= m; cols[c] ^= m; boxes[b] ^= m; grid[i] = 0
        return False

    cells = list(range(81))
    rng.shuffle(cells)
    assert rec(cells)
    return grid


def count_solutions(grid, limit=2):
    rows, cols, boxes, blanks = [0] * 9, [0] * 9, [0] * 9, []
    for i, v in enumerate(grid):
        r, c, b = PEERS[i]
        if v:
            m = 1 << v
            rows[r] |= m; cols[c] |= m; boxes[b] |= m
        else:
            blanks.append((r, c, b))
    found = [0]

    def rec(rem):
        if not rem:
            found[0] += 1
            return
        best, best_n = 0, 10
        for j, (r, c, b) in enumerate(rem):
            n = 9 - bin(rows[r] | cols[c] | boxes[b]).count("1")
            if n < best_n:
                best, best_n = j, n
        r, c, b = rem[best]
        rest = rem[:best] + rem[best + 1:]
        for v in range(1, 10):
            m = 1 << v
            if (rows[r] | cols[c] | boxes[b]) & m:
                continue
            rows[r] |= m; cols[c] |= m; boxes[b] |= m
            rec(rest)
            rows[r] ^= m; cols[c] ^= m; boxes[b] ^= m
            if found[0] >= limit:
                return

    rec(blanks)
    return found[0]


# Nested clue sets like the released test boards: hard's givens are a subset of medium's, medium's of easy's.
def make_board(seed):
    rng = random.Random(seed)
    while True:
        solution = fill(rng)
        puzzle, masks, clues = list(solution), {}, 81
        order = list(range(81))
        rng.shuffle(order)
        for i in order:
            keep, puzzle[i] = puzzle[i], 0
            if count_solutions(puzzle) != 1:
                puzzle[i] = keep
                continue
            clues -= 1
            for name, target in CLUES.items():
                if clues == target:
                    masks[name] = [v != 0 for v in puzzle]
            if clues == min(CLUES.values()):
                return solution, masks


def test_grids(data_dir):
    if not data_dir:
        return set()
    with open(os.path.join(data_dir, "hard", "solutions.pickle"), "rb") as f:
        return {tuple(t for t in s if t != 10) for s in pickle.load(f)}


def main():
    ap = argparse.ArgumentParser(description="generate unique-solution Sudoku boards for training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--test-data-dir", default="data/sudoku", help="grids found here are excluded from the pool")
    a = ap.parse_args()

    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; keeping it")
    exclude = test_grids(a.test_data_dir)
    t0, boards, dropped = time.time(), [], 0
    with multiprocessing.Pool(a.workers) as pool:
        for solution, masks in pool.imap(make_board, range(a.seed, a.seed + a.n), chunksize=16):
            if tuple(solution) in exclude:
                dropped += 1
                continue
            boards.append((solution, masks))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(
        tmp,
        solutions=np.array([b[0] for b in boards], dtype=np.uint8),
        **{name: np.array([b[1][name] for b in boards], dtype=bool) for name in CLUES},
    )
    os.replace(tmp, a.out)
    print(f"{len(boards)} boards ({dropped} test grids dropped) -> {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
