# Contributing to GeoPulse

Thank you for considering a contribution. GeoPulse is a scientific research
code intended to accompany peer-reviewed publications; correctness and
reproducibility come before speed of feature delivery.

## Branch Strategy

**Current model (v0.1.x, alpha) — GitHub flow:**

```
main        ← the one branch; always releasable, protected, tagged versions
feature/*   ← short-lived, squash-merge into main via PR
```

Branch names: `feature/<module>/<short-desc>`, e.g.
`feature/earth/wait-recursion`. Every merge to `main` goes through a PR
with CI green; releases are cut by tagging `main` (`v0.1.0a0`,
`v0.1.0`, `v0.2.0`, …).

**Planned model (v0.5+, community beta) — gitflow:**

Adopt `develop` and `release/*` branches once one of the following
becomes true: (a) three or more concurrent contributors with
in-flight work that regularly conflicts, (b) staged release trains
(nightly RTD against `develop`, stable RTD against `main`), or
(c) support for multiple concurrent versions in production (backports
to old release lines).

Until then, `develop` and `release/*` would add three-way merge
overhead without buying anything.

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

- [ ] Branch is up-to-date with `main` (rebase, not merge).
- [ ] `pre-commit run --all-files` passes (this covers ruff lint,
      ruff-format, mypy, and file-hygiene hooks — see
      [Local Environment Setup](#local-environment-setup)).
- [ ] `pytest tests/ -m "not slow and not benchmark"` passes.
- [ ] New public functions/classes have NumPy-style docstrings.
- [ ] New physics has an analytic or golden-file test.
- [ ] `CHANGELOG.md` entry added.
- [ ] Commit messages follow Conventional Commits.

Every PR requires at least one approving review. Benchmark PRs (anything
touching `tests/benchmarks/` or a claim of published tolerance) require two.

## Local Environment Setup

One-time. Assumes conda is installed.

```bash
git clone https://github.com/shibaji7/geopulse.git
cd geopulse
conda env create -f environment.yml       # creates geopulse-dev
conda activate geopulse-dev
pre-commit install                        # wires .git/hooks/pre-commit
pre-commit run --all-files                # warm caches, verify everything green
```

After this, every `git commit` runs the `.pre-commit-config.yaml` gates:

| Hook | Catches |
|---|---|
| `ruff` (with `--fix`) | Lint issues (auto-fixes many) |
| `ruff-format` | Formatting differences |
| `trailing-whitespace` | Stray trailing spaces |
| `end-of-file-fixer` | Missing terminal newline |
| `check-yaml` / `check-toml` | Config-file syntax errors |
| `check-added-large-files` | Accidental commits > 500 KB |
| `mypy` (on `src/geopulse/`) | Type errors |

If a hook fails, fix the issue (or accept the auto-fix), `git add`, and
re-`git commit`. Bypass in emergencies with `git commit --no-verify` —
but a red hook is usually the hook doing its job.

Keep `geopulse-dev` active whenever you commit; the hook script needs the
env's `pre-commit` on `PATH`. Refresh pinned hook versions occasionally
with `pre-commit autoupdate` (produces one PR bumping the pins).

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

Under the current model, releases are cut directly from `main`:

1. Bump `_version.py` and `pyproject.toml` in a single PR.
2. Add a `[X.Y.Z] - YYYY-MM-DD` entry at the top of `CHANGELOG.md`.
3. Merge to `main`.
4. Tag: `git tag -a vX.Y.Z -m "..."` then `git push origin --tags`.
5. Promote to a GitHub Release: `gh release create vX.Y.Z --notes-from-tag`
   (add `--prerelease` for alphas/betas/RCs).

Once the project adopts gitflow (see [Branch Strategy](#branch-strategy)),
this will switch to `release/vX.Y` stabilisation branches with
cherry-picked fixes.
