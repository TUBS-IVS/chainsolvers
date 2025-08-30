import numpy as np
import logging

from . import helpers as h

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
            s = potentials
        elif potentials is None:
            s = -dist_deviations
        else:
            s = -dist_deviations #TODO: actually use potentials
            # s = pot_weight * potentials - dist_dev_weight * dist_deviations
        return s


class Selector:
    # ---------- normalization helpers ----------
    @staticmethod
    def _log_nonfinite(scores: np.ndarray, where: str) -> None:
        bad = ~np.isfinite(scores)
        if bad.any():
            n_bad = int(bad.sum())
            # show up to 3 examples to avoid log spam
            ex = scores[bad][:3]
            logger.debug(
                f"[Selector:{where}] Detected {n_bad} non-finite score(s) "
                f"(e.g., {ex}). Treating them as missing."
            )

    @staticmethod
    def _normalized_weights(scores: np.ndarray, where: str) -> np.ndarray:
        """
        Shift+scale to [0,1] for probabilities/cutoffs.
        Non-finite -> NaN -> weight 0. If degenerate, return uniform.
        """
        s = np.asarray(scores, dtype=float)
        if logger.isEnabledFor(logging.DEBUG):
            Selector._log_nonfinite(s, where=where)

        # mark non-finite as NaN and ignore in nanmin/nanmax
        s = s.copy()
        s[~np.isfinite(s)] = np.nan

        smin = np.nanmin(s) if np.isnan(s).sum() < s.size else np.nan
        smax = np.nanmax(s) if np.isnan(s).sum() < s.size else np.nan

        if not np.isfinite(smin) or not np.isfinite(smax):
            # all were non-finite -> uniform
            return np.full(s.size, 1.0 / s.size)

        span = smax - smin
        if not np.isfinite(span) or span <= 0.0:
            # all equal finite values -> uniform
            return np.full(s.size, 1.0 / s.size)

        w = (s - smin) / span                 # affine map to [0,1]
        w[~np.isfinite(w)] = 0.0              # NaNs (from non-finite inputs) -> 0 weight
        tot = w.sum()
        if not np.isfinite(tot) or tot <= 0.0:
            return np.full(s.size, 1.0 / s.size)
        return w / tot

    @staticmethod
    def _normalized_nonneg(scores: np.ndarray, where: str) -> np.ndarray:
        """
        Shift+scale to [0,1] but NOT renormalized to sum=1 (useful for cutoffs).
        Non-finite -> 0.
        """
        s = np.asarray(scores, dtype=float)
        if logger.isEnabledFor(logging.DEBUG):
            Selector._log_nonfinite(s, where=where)
        s = s.copy()
        finite = np.isfinite(s)
        if not finite.any():
            return np.zeros_like(s)

        smin = np.nanmin(s)
        smax = np.nanmax(s)
        span = smax - smin
        if not np.isfinite(span) or span <= 0.0:
            out = np.zeros_like(s)
            out[finite] = 1.0   # all equal -> treat as equal 1.0
            return out

        out = np.zeros_like(s)
        out[finite] = (s[finite] - smin) / span
        # non-finite remain 0
        return out

    # ---------- selection ----------
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
        Returns indices (n,) of selected candidates.
        - Scorer may emit any real values (neg/pos, finite/non-finite).
        - This method normalizes safely for probability-based steps and for cutoffs.
        """
        scores = np.asarray(scores, dtype=float)
        assert scores.ndim == 1 and scores.size > 0

        if num_candidates >= scores.size:
            idx = np.arange(scores.size, dtype=np.intp)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Wanted {num_candidates} >= {scores.size}. Selecting all.")
            return idx

        if rng is None:
            rng = np.random.default_rng()

        if strategy == "monte_carlo":
            # shift+scale -> probs
            p = Selector._normalized_weights(scores, where="monte_carlo")
            idx = rng.choice(scores.size, size=num_candidates, p=p, replace=False)

        elif strategy == "top_n":
            # pure ranking uses raw scores (normalization not needed)
            idx = np.argpartition(scores, -num_candidates)[-num_candidates:]
            idx = idx[np.argsort(scores[idx])[::-1]]

        elif strategy == "mixed":
            num_top = int(np.ceil(num_candidates * top_portion))
            num_mc  = num_candidates - num_top

            top_idx = np.argpartition(scores, -num_top)[-num_top:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            if num_mc > 0:
                rest = np.setdiff1d(np.arange(scores.size, dtype=np.intp), top_idx, assume_unique=False)
                if rest.size > 0:
                    p = Selector._normalized_weights(scores[rest], where="mixed_rest")
                    mc_idx = rng.choice(rest, size=num_mc, p=p, replace=False)
                    idx = np.concatenate([top_idx[:num_top], mc_idx])
                else:
                    idx = top_idx[:num_candidates]
            else:
                idx = top_idx[:num_candidates]

        elif strategy == "spatial_downsample":
            assert coords is not None, "coords required for spatial_downsample"
            if num_cells_x is None:
                num_cells_x = max(1, int(np.sqrt(num_candidates)) + 1)
            if num_cells_y is None:
                num_cells_y = num_cells_x
            keep = h.even_spatial_downsample(coords, num_cells_x, num_cells_y)
            idx = np.asarray(keep[:num_candidates], dtype=np.intp)

        elif strategy == "top_n_spatial_downsample":
            assert coords is not None, "coords required for top_n_spatial_downsample"

            # Use shifted+scaled [0,1] for cutoff/ties (robust to all-negative/all-positive/NaN/inf)
            safe = Selector._normalized_nonneg(scores, where="top_n_spatial_downsample")
            sorted_idx = np.argsort(safe)[::-1]

            if num_candidates >= sorted_idx.size:
                return sorted_idx.astype(np.intp, copy=False)

            cutoff = safe[sorted_idx[num_candidates - 1]]
            strictly_better = sorted_idx[safe[sorted_idx] > cutoff]
            ties_at_cutoff  = np.flatnonzero(safe == cutoff)

            need = num_candidates - strictly_better.size
            if need <= 0:
                idx = strictly_better[:num_candidates]
            else:
                num_cells = max(1, int(np.sqrt(need)) + 1)
                keep = h.even_spatial_downsample(coords[ties_at_cutoff], num_cells, num_cells)
                idx = np.concatenate([strictly_better, ties_at_cutoff[np.asarray(keep[:need], dtype=int)]])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return np.asarray(idx, dtype=np.intp)
