# Contributing to GeoPulse

Thank you for considering a contribution. GeoPulse is a scientific research
code intended to accompany peer-reviewed publications; correctness and
reproducibility come before speed of feature delivery.

## Branch Strategy

```
main        ← always releasable, protected, tagged versions
develop     ← integration branch, nightly CI
feature/*   ← short-lived, squash-merge into develop
release/*   ← stabilization branches, cherry-pick fixes only
```

Branch names: `feature/<module>/<short-desc>`, e.g. `feature/earth/wait-recursion`.

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(earth): implement Wait recursion for layered 1-D
fix(solver): correct sign convention in LPM J_e computation
docs(readme): add conda installation instructions
test(earth): add uniform half-space analytic validation
refactor(network): extract Node/Branch dataclasses to base.py
chore(ci): add Python 3.13 to test matrix
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`, `build`, `ci`.

## Pull Request Checklist

Before opening a PR:

- [ ] Branch is up-to-date with `develop`.
- [ ] `ruff check src/ tests/` passes with zero errors.
- [ ] `ruff format --check src/ tests/` passes.
- [ ] `pytest tests/ -m "not slow and not benchmark"` passes.
- [ ] New public functions/classes have NumPy-style docstrings.
- [ ] New physics has an analytic or golden-file test.
- [ ] `CHANGELOG.md` entry added.
- [ ] Commit messages follow Conventional Commits.

Every PR requires at least one approving review. Benchmark PRs (anything
touching `tests/benchmarks/` or a claim of published tolerance) require two.

## Coding Standards Summary

Full rules live in Section 20 of `geopulse_handoff.md`. The essentials:

- **SI units throughout.** Convert nT → T, km → m at data-loader boundaries only.
- **Units in variable names.** `thickness_m`, `conductivity_Sm`, `impedance_Ohm`.
  Never bare `sigma` or `thickness`.
- **Frozen dataclasses** for data containers (`BFieldTimeSeries`, `Node`,
  `Branch`, `SolverResult`, `DeviceResponse`).
- **NumPy-style docstrings** on every public class and method. Parameters,
  Returns, Raises, Notes (with equation + citation), Examples.
- **No `print()`.** Use `from loguru import logger`.
- **No pandas/xarray in core.** Optional extras only (`[data]`).
- **`from __future__ import annotations`** at the top of every `.py` file.
- **`__all__`** list in every module.
- **`pathlib.Path`** not `os.path`.
- **`np.random.default_rng(seed)`** not `np.random.rand()`.
- **No bare `except:`.** Always catch specific exceptions.
- **Cite the equation source** (Author, Year) in every physics docstring.
- Flag design questions to the PI with `# TODO(PI): <question>` rather than
  guessing silently.

## Testing

Tests mirror `src/` structure. `src/geopulse/earth/layered_1d.py` →
`tests/unit/test_earth/test_layered_1d.py`.

Every physics function needs an analytic or reference-implementation validation
(e.g., uniform half-space matches `Z = sqrt(iωμ₀/σ)` at `rtol=1e-12`).

## Release Process

Releases are cut from `release/vX.Y` branches. See Section 19 of the handoff
spec for the full procedure.
