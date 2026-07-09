# Manuscript v2 — presentation pass (+ substantive-fix work)

This folder holds a **presentation-focused rewrite** of the disc_SLV manuscript. It is a
copy of `../disc_SLV.tex` (the v1 manuscript) with the prose being de-densified section by
section. **Scientific claims are unchanged in the presentation pass** — only wording, sentence
structure, and emphasis.

**Substantive-fix work (separate from the presentation pass):**
- **Concern #1 — out-of-sample SSR-forecast test (RUN on real data):** design at
  [`ssr_forecast_test_design.md`](ssr_forecast_test_design.md); harness + passing synthetic
  self-test in [`scripts/`](scripts/); **real-data run on ORATS SPX EOD 2015–2023 via
  `scripts/wire_orats.py`, findings in [`ssr_forecast_findings.md`](ssr_forecast_findings.md).**
  Result: the **honest null** — the option-implied (Doeff–Kamal skew-decay) SSR is near-constant
  (~1.7) and does **not** forecast next-period realised comovement better than a trailing realised
  estimate (encompassing null; negatively correlated and worse-than-Black hedging in the grind).
  A *stable* SSR does beat a *trailing-realised* delta in normal regimes, so the fixed-R replay is
  defensible — but the edge is stability, not forward-looking prediction. §8's "listed below"
  promise should be filled with this (a pending paper-text edit). Full-model F^Q wiring
  (`scripts/wire_poc.py`) is a heavier variant that would give the same qualitative answer.
- **Concern #4 — level-independent smile-shape limitation (DONE, in `disc_SLV_v2.tex`):** the
  "shape sticks" result was framed only as a feature. Added (a) a caveat at the end of the §8
  "level rides, the shape sticks" paragraph naming it a *structural constraint* of translation
  invariance (conditional smile depends on the vol regime, not the spot level; deterministic
  spot-level-dependent shape is out of reach — only the indirect stochastic-regime channel), and
  (b) a second "structural limit" row in the §9 ledger (Table 6).
- Remaining: multi-month level-bias caveat into the paper text; the real-data run of the Concern #1 test.

- Source: `disc_SLV_v2.tex`
- Build: `latexmk -pdf disc_SLV_v2.tex` (self-contained; inline `thebibliography`, no `.bib`).
  Figures resolve via `\graphicspath{{../}}` → `../figs/`.
- Status: compiles clean (0 errors, 0 undefined refs; 40 pp).

> **⚠ v1/v2 substance divergence (since Concern #4).** Until now v2 = v1 content + presentation.
> The Concern #4 caveat + ledger row are **v2-only** substantive additions (v1 does not have them),
> so the token/word parity checks below now show v2 *ahead* of v1 by these additions — expected, not
> a defect. Recommendation: treat **v2 as canonical** and fold any future v1 edits *into* v2 (not the
> reverse), or the Concern #4 additions must be carried back manually. The parity check still catches
> *v1* advancing (its tokens would appear as missing from v2).

## Status: COMPLETE (abstract + intro + full body)
Compiles clean: 0 errors, 0 undefined refs, 4 overfull boxes (below the v1 baseline), 39 pp.
Verified against v1: `\cite/\ref/\eqref/\label/\bibitem` multiset **identical**, and body
content-word count matches to within 1 word — no claim, number, or reference altered or dropped.

### Pass 1 — abstract + intro
- **Abstract** — was one ~230-word block; now four short paragraphs, one idea per sentence,
  decorative `\emph{}` stripped.
- **§1 Introduction** — full rewrite. Multi-em-dash run-ons split into single-idea sentences;
  contributions made parallel; related work split in two. All refs/cites/equation/table preserved.

### Pass 2 — full body (§2–§10 + appendices)
- Worst run-ons split throughout (§4.6 fusion, §6 SSR, §8 de-eventing + hedging, §9 discussion,
  §10 conclusion, Appendix C intro). Math, equations, algorithms, tables, and proofs untouched.
- **Coinages resolved** (see table below): retired *rung / ladder / Sobolev-ladder* (§5 title now
  "The fitting step: digitals as a price metric"; `\label{sec:rung}` kept for stable cross-refs);
  glossed *fibre* (§4.3) and *clean flanks* (§7.1); neutralized *benign* → "affects only the level"
  (§4.6 prose) and the ledger row → **leverage collocation** (§9 Table).

### Reconciliation with v1 concurrent edits (2026-07-07)
v1 (`../disc_SLV.tex`) was edited at 20:21, after this v2 forked at 20:09 — a concurrent session
added honesty caveats responding to the review. **Four** token-bearing additions were ported into v2:
1. Hedging (§8): "Two honest readings temper the headline…" (min-variance-delta framing + missing
   realised-SSR benchmark). Semicolons reflowed to periods for v2 consistency.
2. Discussion (§9): "Three further caveats…" (stress-regime, least-identified two-timescale block,
   SANOS-preprint dependency). Kept as a parallel semicolon list.
3. Intro (§1): "…a deliberately weaker commitment: more portable, less tightly identified…".
4. `\bibitem{hullwhite2017}` (Hull–White 2017, optimal delta hedging).

**Hazard note:** `disc_SLV.tex` (v1) is edited by concurrent sessions and is not under git. Before
any future re-sync, re-run the token-parity + word-count checks (below) — do NOT assume v1 is static.

## Rewrite principles (applied throughout; use for any future edits)
1. One idea per sentence. A sentence with 3+ em-dash clauses gets split.
2. Em-dashes for genuine asides only, not as a default connective.
3. `\emph{}` only on a term of art at first use; remove decorative emphasis.
4. Cut hedge/filler qualifiers ("cleanly", "genuine", "above all", "by construction" when redundant).
5. Never change a number, claim, `\cite`, `\ref`, label, or equation. Presentation only.
6. After each section, recompile and check `Overfull` count has not risen above the v1 baseline.

## Coinage decisions — APPLIED
  | coinage | decision applied | where |
  |---|---|---|
  | "fibre" | kept; glossed once ("the kernel's conditional law at a fixed state $(z,f,s)$") | §4.3 |
  | "clean flanks" / "clean-flank bridge" | kept; glossed ("the *clean flanks* that name the method") | §7.1 |
  | "Sobolev ladder" / "fitting rung" / "digital rung" | retired → "fitting step" / "digital level" / "digital fit"; Sobolev `W^{1,1}` math kept | §5 title, §4.7, §5, §7.2 |
  | "SANOS ladder" / "convex-order ladder" | retired → "chain" | App C |
  | "benign" (collocation) | neutralized → "affects only the level"; ledger row → "leverage collocation" | §4.6, §9 Table |
  | "elasticity-scaled Jacobian" | kept (precise, defined in App D) | — |
  | "the desk receives / lands on the desk" | left as-is (acceptable once; low priority) | §5, §7 |

## Verification commands (re-run before any re-sync with v1)
```sh
# token parity (should print IDENTICAL):
extract(){ grep -oE '\\(cite|ref|eqref|label|bibitem)\{[^}]*\}' "$1"|sort|uniq -c; }
diff <(extract ../disc_SLV.tex) <(extract disc_SLV_v2.tex)
# body word-count parity (should be within a few words):
# python3 strip-to-body-words on each; a gap = unported net content
```

## Notes
- `manuscript_v2/` is deliberately separate from `../v2/`, which holds the empirical/POC v2
  (data, findings HTML/PDF), not the manuscript.
- Not under version control (repo is not a git repo); the v1 source at `../disc_SLV.tex` is untouched.
