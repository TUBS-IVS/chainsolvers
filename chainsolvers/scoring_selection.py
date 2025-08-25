import numpy as np
import helpers as h
import logging
logger = logging.getLogger(__name__)

"""
Scoring and selection assumes the general layout of several equal-len np arrs, holding e.g. ids, coords, potentials;
though it is agnostic about what they actually hold as long as they are equal-len np arrays. 
The Selector thus just returns indices.
"""

class Scorer:
    @staticmethod
    def score(
        *,
        potentials: np.ndarray | None = None,
        dist_deviations: np.ndarray | None = None,
        pot_weight = 1.0,
        dist_dev_weight = 1.0,
    ) -> np.ndarray:
        """
        Return raw (non-normalized) scores, 1D (k,), negative to positive infinity.

        - If only dist_deviations is given: score = dist_deviations.
        - If only potentials is given: score = potentials.
        - If both provided: combine additively (you can swap to weighted if desired).
        """
        if potentials is None and dist_deviations is None:
            raise ValueError("Need at least one of potentials or dist_deviations.")
        if dist_deviations is None:
            s = potentials.astype(float, copy=False)
        elif potentials is None:
            s = dist_deviations.astype(float, copy=False)
        else:
            s = pot_weight * potentials.astype(float, copy=False) + dist_dev_weight * dist_deviations.astype(float, copy=False)
        return s

class Selector:
    @staticmethod
    def select(
        scores: np.ndarray,
        num_candidates: int,
        strategy: str = "monte_carlo",
        *,
        coords: np.ndarray | None = None,
        top_portion: float = 0.5,
        num_cells_x: int | None = None,
        num_cells_y: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Returns (indices (n,), chosen_scores (n,)).
        """
        assert scores.ndim == 1 and scores.size > 0
        if num_candidates >= scores.size:
            idx = np.arange(scores.size)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Wanted candidates is larger than given candidates. Selecting all {scores.size} candidates")
            return idx

        if rng is None:
            rng = np.random.default_rng()

        if strategy == "monte_carlo":
            total = float(scores.sum())
            if not np.isfinite(total) or total <= 0.0:
                # Fallback to uniform if scorer gave all-zeros/non-finite
                p = np.full(scores.size, 1.0 / scores.size)
            else:
                p = scores / total
            idx = rng.choice(scores.size, size=num_candidates, p=p, replace=False)

        elif strategy == "top_n":
            idx = np.argpartition(scores, -num_candidates)[-num_candidates:]
            idx = idx[np.argsort(scores[idx])[::-1]]

        elif strategy == "mixed":
            num_top = int(np.ceil(num_candidates * top_portion))
            num_mc  = num_candidates - num_top
            top_idx = np.argpartition(scores, -num_top)[-num_top:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            rest = np.setdiff1d(np.arange(scores.size), top_idx, assume_unique=False)
            if num_mc > 0 and rest.size > 0:
                rest_scores = scores[rest]
                total = float(rest_scores.sum())
                p = (rest_scores / total) if total > 0 else np.full(rest.size, 1.0 / rest.size)
                mc_idx = rng.choice(rest, size=num_mc, p=p, replace=False)
                idx = np.concatenate([top_idx[:num_top], mc_idx])
            else:
                idx = top_idx[:num_candidates]

        elif strategy == "spatial_downsample":
            assert coords is not None, "coords required for spatial_downsample"
            if num_cells_x is None:
                num_cells_x = max(1, int(np.sqrt(num_candidates)) + 1)
            if num_cells_y is None:
                num_cells_y = num_cells_x
            keep = h.even_spatial_downsample(coords, num_cells_x, num_cells_y)
            idx = np.array(keep[:num_candidates], dtype=int)

        elif strategy == "top_n_spatial_downsample":
            assert coords is not None, "coords required for top_n_spatial_downsample"
            sorted_idx = np.argsort(scores)[::-1]
            cutoff = scores[sorted_idx[num_candidates - 1]] if sorted_idx.size >= num_candidates else scores[sorted_idx[-1]]
            top_indices = np.where(scores >= cutoff)[0]
            if top_indices.size > num_candidates:
                num_cells = max(1, int(np.sqrt(num_candidates)) + 1)
                keep = h.even_spatial_downsample(coords[top_indices], num_cells, num_cells)
                idx = top_indices[np.array(keep[:num_candidates], dtype=int)]
            else:
                idx = sorted_idx[:num_candidates]

        else:
            raise ValueError("Unknown strategy")

        return idx