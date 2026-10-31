import os, pickle

import numpy as np

SEP, BOS = 10, 11
# Token index of each of the 81 cells in the 89-token target (rows joined by separators).
CELL_POS = np.array([r * 10 + c for r in range(9) for c in range(9)])


def _group_perm(rng):
    return np.concatenate([g * 3 + rng.permutation(3) for g in rng.permutation(3)])


def to_tokens(cells):
    tokens = np.full((cells.shape[0], 89), SEP, dtype=np.int64)
    tokens[:, CELL_POS] = cells
    return tokens


class SudokuBoards:
    """Samples (puzzle, solution) tokens from a gen_sudoku.py pool, with validity-preserving symmetries."""

    def __init__(self, path, difficulties, augment=True):
        with np.load(path) as z:
            self.solutions = z["solutions"].astype(np.int64)
            self.masks = [z[d] for d in difficulties]
        self.augment = augment

    def _augment_one(self, grid, mask, rng):
        rows, cols = _group_perm(rng), _group_perm(rng)
        grid, mask = grid.reshape(9, 9)[rows][:, cols], mask.reshape(9, 9)[rows][:, cols]
        if rng.random() < 0.5:
            grid, mask = grid.T, mask.T
        digits = np.concatenate([[0], rng.permutation(9) + 1])
        return digits[grid].reshape(-1), mask.reshape(-1)

    def sample(self, batch_size, rng):
        grids, masks = [], []
        for i in rng.integers(0, self.solutions.shape[0], batch_size):
            grid = self.solutions[i]
            mask = self.masks[rng.integers(0, len(self.masks))][i]
            if self.augment:
                grid, mask = self._augment_one(grid, mask, rng)
            grids.append(grid)
            masks.append(mask)
        grids, masks = np.stack(grids), np.stack(masks)
        puzzle = np.full((batch_size, 91), BOS, dtype=np.int64)
        puzzle[:, 1:90] = to_tokens(grids * masks)
        return puzzle, to_tokens(grids)


def load_test_boards(data_dir, difficulty, n=0):
    with open(os.path.join(data_dir, difficulty, "puzzles.pickle"), "rb") as f:
        puzzles = pickle.load(f)
    with open(os.path.join(data_dir, difficulty, "solutions.pickle"), "rb") as f:
        solutions = pickle.load(f)
    if n:
        puzzles, solutions = puzzles[:n], solutions[:n]
    return np.array(puzzles, dtype=np.int64), np.array(solutions, dtype=np.int64)
