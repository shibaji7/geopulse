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

**Allowed types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
`test`, `build`, `ci`, `chore`, `revert`.

**Title length:** ≤ 80 characters. Anything longer belongs in the body.
(The 8-char headroom over the classic 72-char convention lets GitHub's
squash-merge `(#N)` suffix land without tripping the check.)

**Enforcement — two levels:**

- **Locally**, the `gitlint-soft` hook (see [Guardrail hooks
  survival guide](#guardrail-hooks--plain-english-survival-guide))
  runs on `commit-msg` and **warns** on non-conforming titles but
  never blocks. This lets first-time contributors commit quickly
  without wrestling `type(scope): summary` mechanics.
- **In CI**, the `Lint` job in
  [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs
  `gitlint --commits base..HEAD` in **hard** mode on every PR. A
  malformed title on any branch commit blocks the merge. Fix by
  amending / rebasing the offending commit, then re-push.

Fixup commits — `git commit --fixup=<sha>` — are ignored by both
hooks (they're auto-squashed on rebase). Merge, revert, and squash
commits are ignored too. So you only need to think about titles for
"real" commits.

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

| Hook | Stage | Catches |
|---|---|---|
| `ruff` (with `--fix`) | pre-commit | Lint issues (auto-fixes many): style, imports, naming, complexity, docstring format, **type annotations on public functions** |
| `ruff-format` | pre-commit | Formatting differences |
| `trailing-whitespace` | pre-commit | Stray trailing spaces |
| `end-of-file-fixer` | pre-commit | Missing terminal newline |
| `check-yaml` / `check-toml` | pre-commit | Config-file syntax errors |
| `check-added-large-files` | pre-commit | Accidental commits > 500 KB |
| `mypy` (on `src/geopulse/`) | pre-commit | Type errors |
| `interrogate` | pre-commit | Docstring coverage < 85 % anywhere in `src/geopulse/` |
| `pylint-duplicate-code` | pre-commit | Copy-pasted blocks ≥ 15 lines inside `src/geopulse/` (only that; nothing else pylint would flag) |
| `check-no-version-drift` | pre-commit | Live version literal outside `_version.py` / `CHANGELOG.md` |
| `gitlint-soft` | commit-msg | **Warn only.** Nudges the commit message toward Conventional Commits; never blocks the commit locally (CI enforces — see below) |
| `clean-build-artefacts` | pre-commit | Strips `dist/`, `build/`, cache dirs from the working tree |
| `build-wheel-check` | **pre-push** | `python -m build` still succeeds (catches packaging breakage) |
| `build-docs-check` | **pre-push** | `sphinx-build` still succeeds |

Enable the `commit-msg` and `pre-push` hooks the first time:

```bash
pre-commit install --hook-type commit-msg     # for gitlint-soft
pre-commit install --hook-type pre-push       # for the wheel/docs smoke checks
```

If a hook fails, fix the issue (or accept the auto-fix), `git add`, and
re-`git commit`. Bypass in emergencies with `git commit --no-verify` —
but a red hook is usually the hook doing its job, and CI runs the same
checks so the bypass just moves the failure downstream.

### Guardrail hooks — plain-English survival guide

Extra background on the non-obvious ones:

**`ruff ANN` rules — public-function type annotations.** GeoPulse
claims `Typing :: Typed` in `pyproject.toml` classifiers and ships a
`py.typed` marker; the `ANN` rules enforce that every public function
in `src/geopulse/` actually carries type hints for its arguments and
return type. **`ANN401` (bans `typing.Any`) is project-wide ignored**
because a handful of patterns genuinely need it: `dict[str, Any]`
config payloads, `**kwargs: Any` passthrough, and the `Uncertain[T]`
generic wrapper's sample-store. **Stub modules** (files that only
raise `NotImplementedYetError`) are per-file exempted in `ruff.toml`
— when the real implementation lands, drop the exemption in the
same PR. Downstream users benefit from complete type hints for IDE
completion and Sphinx autodoc type inference.

**`interrogate` — docstring coverage.** Fails if the percentage of
docstring-carrying public functions/classes/methods in `src/geopulse/`
drops below **85 %**. On failure it prints a per-file table showing
exactly which files pulled the average down; add the missing docstring
and re-commit. Do NOT add `# noqa` — write the docstring, it's cheap
and load-bearing for RTD.

**`pylint-duplicate-code` — copy-paste detector.** Fails if it finds
≥ 15 consecutive lines that appear in two places inside
`src/geopulse/`. Scoped tight: doesn't check tests (they *should*
share fixtures), doesn't check examples (case-studies legitimately
share ingest boilerplate). If it fires, refactor the shared block
into a helper — usually a small function or dataclass method. The
threshold (15) tolerates the natural π-section assembly patterns
that repeat between network subclasses; if you legitimately need to
share more, put it in a shared helper module.

**`check-no-version-drift`.** Fails if the current live version
(read from `_version.py`) appears anywhere except `_version.py`
itself and `CHANGELOG.md`. See the [Version-drift
guardrail](#version-drift-guardrail-check-no-version-drift) section
below for the full policy.

**`check-docstring-drift` — docstring/signature consistency.** Fails if
a function's numpy-style docstring makes a claim the code no longer
backs up: a `Parameters` section that lists an argument no longer in
the signature (or is missing one that is), or a `Raises` section that
doesn't mention an exception the function body actually raises. Only
checks functions whose docstring **already has** a `Parameters` or
`Raises` section — it never demands one exist (that's `interrogate`'s
job), so it stays quiet on the numpy-style-with-type-hints and
`Attributes`-on-dataclasses conventions that tripped up pydoclint and
numpydoc-validation (see [GH-17](https://github.com/shibaji7/geopulse/issues/17)
for the evaluation). On failure it prints the file, line, and function
name for each mismatch — update whichever side (code or docstring) is
wrong and re-commit.

**`gitlint-soft` — commit message.** Runs on `commit-msg`. Prints a
warning if the commit title doesn't match Conventional Commits
(`type(scope): summary`, ≤ 80 chars, one of the allowed types),
**but never blocks the commit locally.** CI runs the same check in
HARD mode and *will* block the PR merge if any commit on the branch
has a malformed title — so the local warning is a friendly heads-up,
not a wall. See [Commit Conventions](#commit-conventions).

Keep `geopulse-dev` active whenever you commit; the hook script needs the
env's `pre-commit` on `PATH`. Refresh pinned hook versions occasionally
with `pre-commit autoupdate` (produces one PR bumping the pins).

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

## Versioning Conventions

GeoPulse follows [Semantic Versioning 2.0](https://semver.org/): a
release is `MAJOR.MINOR.PATCH`, optionally suffixed with a Python
pre-release identifier (`aN` / `bN` / `rcN`).

**Rule of thumb** — when your PR lands, ask: *does this force a
downstream user to change their code?*

| Change | Bump | Example |
|--------|------|---------|
| Bug fix; no API change | `PATCH` | `0.2.0` → `0.2.1` |
| New feature, existing API untouched | `MINOR` | `0.2.1` → `0.3.0` |
| Breaking API change (renamed function, changed signature, removed public name) | `MAJOR` | `0.9.3` → `1.0.0` |
| Docs / infra only (README, CI, tests) | No bump; batch with the next release |

Zero-major (`0.x.y`) is treated as "public API can still break"; we
still try to bump `MAJOR` (i.e. `0.x` → `0.(x+1)`) rather than silently
break things, but there's no promise. Once we ship `1.0.0`, `MAJOR`
means real breakage.

### Pre-release tags (`aN` / `bN` / `rcN`)

Between two `.0` releases, we cut pre-releases so consumers can pin.
Same tag mechanism, different semantics:

| Suffix | Meaning | API stability | Purpose |
|--------|---------|---------------|---------|
| `aN` (**alpha**) | `0.2.0a1`, `0.2.0a2`, … | Not stable — breaking changes allowed between alphas | Testing package plumbing (build, publish, CI, docs); early integrator feedback |
| `bN` (**beta**) | `0.2.0b1`, `0.2.0b2` | Frozen for the target release — no new features after `b1` | Integration testing, real-storm shakedowns, bug hunting |
| `rcN` (**release candidate**) | `0.2.0rc1` | Frozen — critical fixes only | Final testing before the `.0` cut |
| *(no suffix)* | `0.2.0` | Stable public API for that MINOR line | Ship to PyPI + GitHub Release |

Rules of thumb for pre-release bumps:

- **`aN` → `a(N+1)`** — any change during alpha. Alphas are cheap; use
  them liberally when work-package PRs land.
- **`a` → `b1`** — declaration that the MINOR is now feature-frozen.
  Requires: ROADMAP row for this MINOR reads all ✅; docs updated;
  CHANGELOG has a full `[X.Y.0b1]` entry.
- **`b` → `rc1`** — declaration of feature freeze *including bug
  content*. Only fixes for regressions found in beta land after
  `rc1`. Requires: no open blocker issues tagged for this MINOR.
- **`rc` → `.0`** — no changes since `rcN` other than the version
  bump. If new commits landed, cut another `rc(N+1)` instead.

### Deciding what a change is

Some judgement calls:

- **Adding a new keyword argument with a default** — MINOR
  (backward-compatible additive).
- **Renaming a public class** — MAJOR (even if the old name still
  works via a shim, remove the shim in the next MAJOR).
- **Changing default value of an argument** — MAJOR if the change
  produces different numbers; MINOR if it only changes ergonomics
  (e.g. figure size).
- **Adding a new module** — MINOR.
- **Changing a physics implementation so results shift by more than
  the stated tolerance** — MAJOR, even if the API is unchanged.
  Reproducibility is part of the public contract.
- **Loosening or tightening the pinned dependency floors in
  `pyproject.toml`** — MINOR (floor bump); MAJOR only if it drops
  support for a Python version.

If unsure, describe the change in the PR description and ask a
maintainer to confirm the bump. Better to over-communicate than to
silently ship a `MINOR` that should have been `MAJOR`.

### Where the version number lives

**Single source of truth:** `src/geopulse/_version.py` — the string
literal assigned to `__version__` is the version. Nothing else
duplicates it.

- `pyproject.toml` declares `dynamic = ["version"]` and reads
  `__version__` at build time via `[tool.setuptools.dynamic]`. **Do
  not** hand-edit a version string there — there is none to edit.
- `docs/conf.py` imports `__version__` directly (no fallback: if the
  import breaks, the Sphinx build fails loudly rather than publishing
  a stale label to RTD).
- `CHANGELOG.md` — add a `[X.Y.Z-suffix] - YYYY-MM-DD` header at the
  top of the file when you bump. Historical entries stay forever
  (they're the release archive).

### Version-drift guardrail (`check-no-version-drift`)

A pre-commit hook + CI step runs
[`scripts/check_no_version_drift.py`](scripts/check_no_version_drift.py)
on every commit and every pipeline run. It fails if the current live
version literal (read from `_version.py`) appears anywhere except
`_version.py` itself or `CHANGELOG.md`.

**Why:** it's easy to copy the current version into a docs example
("try `pip install geopulse==0.2.0a1`") and then forget to update
that example on the next bump. The hook catches that at commit time.

**Where it runs:**

- **Pre-commit** — `.pre-commit-config.yaml` → `check-no-version-drift`
  in the `pre-commit` stage. Blocks the commit if any file in
  `README.md`, `ROADMAP.md`, `docs/**/*.md`, `docs/**/*.rst`,
  `docs/conf.py`, or `pyproject.toml` contains the live version.
- **CI** — same script runs as an extra step in the `Lint (ruff +
  mypy)` job in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
  Catches anyone who bypassed pre-commit with `--no-verify`.

**When it fires, you'll see:**

```
check-no-version-drift: found 1 hardcoded live-version literal(s)
  README.md:142: geopulse-0.2.0a1 is the current release

The live version '0.2.0a1' must appear only in _version.py and
CHANGELOG.md. Either remove the literal above, or add the file to
the ALLOWED set in scripts/check_no_version_drift.py with a comment
explaining why the waiver is safe.
```

**How to fix:** almost always, rewrite the offending line to not
name a specific version. Preferred patterns:

- Instead of `pip install geopulse==0.2.0a1` → `pip install geopulse`
  (users get whatever PyPI serves as latest).
- Instead of `geopulse 0.2.0a1 supports …` → `GeoPulse supports …`
  (drop the version qualifier — the current version is on the PyPI
  badge above the fold).
- Instead of a hardcoded wheel URL → point at
  `/releases/latest` and let the user pick.

**When a waiver is genuinely needed:** add the file path to the
`ALLOWED` set at the top of
[`scripts/check_no_version_drift.py`](scripts/check_no_version_drift.py)
in the *same PR* that introduces the literal, with an inline comment
explaining why the literal must live there and why keeping it
in-sync manually is acceptable. Adding to `ALLOWED` is a real
decision; think of it like adding a `# noqa` — sometimes right,
never accidental.

**Limitations:** the check only defends against the *current* live
version leaking. It does *not* catch someone hardcoding a *different*
version string (e.g. writing `0.5.0` into README while `_version.py`
is still `0.2.0a1`). Catching that would need per-line waiver
markers to avoid false-positives on `ROADMAP.md` milestone labels
and `CONTRIBUTING.md` illustrative examples. If that becomes a real
problem, extend the script with a `<!-- version-check: allow -->`
inline convention.

## Release Process

Under the current model, releases are cut directly from `main`:

1. Bump the version in `src/geopulse/_version.py` (single-line edit)
   — pick the next tag per the rules in
   [Versioning Conventions](#versioning-conventions). Nothing else
   needs to be touched: `pyproject.toml` reads dynamically at build
   time, `docs/conf.py` reads at Sphinx-build time.
2. Add a `[X.Y.Z] - YYYY-MM-DD` entry at the top of `CHANGELOG.md`
   summarising the release.
3. Merge the bump PR to `main`.
4. Tag: `git tag -a vX.Y.Z -m "..."` then `git push origin --tags`.
5. Promote to a GitHub Release: `gh release create vX.Y.Z --notes-from-tag`
   (add `--prerelease` for alphas/betas/RCs).

The OIDC-based publish workflow at
[`.github/workflows/release.yml`](.github/workflows/release.yml)
picks up the `v*` tag, builds sdist + wheel (both stamped with the
version read from `_version.py`), and publishes to PyPI without any
long-lived API tokens.

Once the project adopts gitflow (see [Branch Strategy](#branch-strategy)),
this will switch to `release/vX.Y` stabilisation branches with
cherry-picked fixes.
