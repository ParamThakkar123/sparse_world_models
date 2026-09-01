# Pre-submission runbook — ICLR 2027

Everything to check before the paper goes to OpenReview, in the order it has to happen.
Written 2026-08-18. Policy facts are quoted from the ICLR 2027 pages as fetched that day and
are **re-verified in step 0** — this project's LaTeX history records two prior misreadings of
a call for papers, so no policy line here is to be trusted at submission time without opening
the source.

| what | when | source |
|---|---|---|
| **Abstract** (mandatory) | **2026-09-18 AOE** | [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) |
| **Full paper** | **2026-09-25 AOE** | [Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers) |
| Reviews released | 2026-11-05 | CFP |
| Author–reviewer discussion | 2026-11-05 → 11-18 | CFP |
| Decisions | 2026-12-16 | CFP |

Main text **9 pages or fewer** at submission (10 during discussion and camera-ready).
References and appendices do not count. AOE = UTC−12.

---

## Step 0 — Do this first, four weeks out (by ~2026-08-25)

These are the items that can invalidate the submission no matter how good the paper is.

- [ ] **Re-read all four policy pages verbatim.** Author Guidelines, Call for Papers,
      [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors), and the
      style-file `instructions`. Do not rely on this table.
- [ ] **Confirm reciprocal-reviewing eligibility.** ICLR 2027: *"Each author may appear as a
      co-author on at most one paper in which no author is an eligible reciprocal reviewer"*,
      and eligibility means *"at least one accepted publication at one of the following
      venues: ICLR / NeurIPS / ICML / UAI / AISTATS / JMLR / TMLR"* and similar — **primary
      conference papers only; a workshop paper does not count.** The WORLDS @ IROS 2026
      workshop paper therefore does **not** make anyone eligible. Also: *"all submissions must
      have at least one author who is registered to review at least 3 papers"*, and failure to
      deliver those reviews *"may have their paper submissions desk rejected"*.
      → **Register to review, at the right load, before the abstract deadline.**
- [ ] **Confirm the workshop paper does not trip dual submission.** The policy says papers
      *"that have been presented at workshops (i.e., venues that do not have publication
      proceedings) do not violate the policy"*. Confirm WORLDS @ IROS 2026 was
      **non-archival**. If it had proceedings, this submission needs substantial new content
      framed as such — which it has, but the framing has to be explicit.
- [ ] **Decide the framing, in writing.** The evidence supports the degeneracy/methodology
      paper (three benchmarks, three one-liners, five published models, the audit battery).
      It does not currently support a "corrected benchmark" paper. Do not start writing
      before this is settled; the two papers share almost no structure.

## Step 1 — The claims, before any prose

- [ ] Read `experiments/RESULTS.md` → **Status map**. It is the only place that says which
      claims are current. Anything marked *superseded*, *refuted*, *retracted* or
      *de-emphasised* must not appear as a claim, only as a recorded refutation.
- [ ] Build a claim ledger: every number in the paper, with the RESULTS.md section and the
      `experiments/runs/` directory it comes from. Anything that cannot be traced is cut.
- [ ] **Nothing from the leaky splits.** Only figures recomputed on `splits_clean_*` may be
      quoted. See "Known issue: the splits behind every number below leak".
- [ ] Confirm each headline comparison quotes a **difference CI**, not two overlapping error
      bars (`experiments/statistics.py`).
- [ ] Confirm no comparison is reported as significant on the strength of `p = 0.250` — that
      is the *floor* at n=3, not a null result. Five-seed reruns land via
      `experiments/iclr_closeout.sh` stage 2; if they did not finish, say n=3 and quote
      effect sizes only.

## Step 2 — Results that still have to land

Tracked in `experiments/iclr_closeout.sh`. Check each off or convert it into an explicit
limitation sentence in the paper. **A stage that did not run becomes a written limitation —
it never becomes silence.**

- [ ] **Stage 1 — clutter, all three filters.** The one result that could change the paper's
      story. If the battery loses on the dense-packed clutter onset benchmark, the paper has a
      *positive* result (a benchmark that resists the shortcuts) and must be rewritten around
      it. If the battery wins again, the "pervasive" claim gains its strongest domain.
- [ ] **Stage 2 — five seeds.** Drops the significance floor from 0.250 to 0.0625.
- [ ] **Stage 3 — three planning seeds.** Until this lands, the paper may say *four of five
      published models plan* and *the sparse model is not distinguishable from the best of
      them*. It may **not** say PETS beats it.
- [ ] **Stage 4 — pixels.** Either a validated number, or the measured negative
      ("standard object-centric front ends fail on sparse-foreground manipulation scenes;
      objects cover ~0.7% of the frame"), which is publishable as a limitation with a cause.
      **No pixel number may be quoted without a completed run.**

## Step 3 — Anonymity (this is the desk-reject step)

*"Any paper where author identity is revealed in either the main text or the supplementary
material will be desk rejected."*

- [ ] `\iclrfinalcopy` is **commented out** in `paper/main.tex`.
- [ ] No author names, affiliations, emails, or funding acknowledgements in the PDF.
- [ ] `pdfinfo paper/main.pdf` — no author or producer string carrying a name.
- [ ] Grep the source: `grep -rniE "thakkar|<affiliation>|gmail" paper/`
- [ ] **The repository link, if included, is anonymised** (anonymous.4open.science or an
      anonymised mirror). A GitHub URL with a username is an identity reveal.
- [ ] If code is submitted as supplementary: strip `.git/`, and check that
      `IEEE_Conference_Template/` — which contains the **non-anonymous workshop paper and its
      PDF** — is excluded. This is the single most likely way this project deanonymises itself.
- [ ] Cite the workshop paper in the **third person**, as prior work by someone else would be
      cited. Related arXiv/workshop papers by the same authors do not break anonymity if cited
      this way.
- [ ] No "our previous paper", "as we showed in", or repo-name giveaways in figure captions.

## Step 4 — Required sections

- [ ] **LLM-use disclosure — mandatory, and does not count toward the page limit.** ICLR 2027
      requires disclosure both on the submission form and in a section of the paper. Disclosure
      is *required* for, among others: *"implement methods"*, *"design or provide feedback on
      research methodology or experiments"*, *"interpret results"*, *"clean and reformat
      dataset"*. **All four apply to this project** — the audit battery, the four-engine suite,
      the published baselines and much of the analysis were implemented with an AI coding
      assistant. Write it plainly and completely; under-disclosure is the risk, not
      over-disclosure. Note also: *"a substantial falsehood... produced by an LLM would be
      considered a Code of Ethics violation on the part of the paper's authors"* — the claim
      ledger in step 1 is what discharges that responsibility.
- [ ] **Reproducibility statement** (strongly encouraged): one paragraph pointing at
      `experiments/RESULTS.md`, the per-experiment commands, and the audit battery.
- [ ] Ethics statement — optional; not obviously needed here.

## Step 5 — The paper mechanically

- [ ] Compiles clean from `paper/` with the **unmodified** `iclr2027_conference.sty`. No
      margin, font-size, or spacing hacks to fit the limit.
- [ ] Main text ≤ 9 pages, measured **after** references start.
- [ ] Every figure legible in greyscale and at print size; no screenshots of tables.
- [ ] All references have venue and year; check the five published baselines' citations
      (Sanchez-Gonzalez 2020, Li 2019, Kipf/van der Pol/Welling ICLR 2020, Wu et al. ICLR 2023,
      Chua et al. NeurIPS 2018, Goyal et al. NeurIPS 2021) against the real bibliographies.
- [ ] `python paper/preflight.py` passes (page count, anonymity grep, PDF metadata, style-file
      integrity).

## Step 6 — Artifact

- [ ] `pytest tests/` green, and the count stated in the paper matches the actual count.
- [ ] `python -m experiments.audit_battery --help` runs, and the README's third-party usage
      example is copy-pasteable.
- [ ] The audit battery is the deliverable — make sure a reader can run it on **their**
      benchmark without touching our data pipeline. `tests/test_audit_battery.py` pins it
      against the published experiment so the two cannot drift.
- [ ] Regeneration commands in `README.md` / `dataset_commands.md` are current.

## Step 7 — Submission day (do not leave to 2026-09-25)

- [ ] Abstract submitted by **2026-09-18 AOE** — mandatory, and a missed abstract means the
      paper cannot be submitted at all.
- [ ] Reviewer registration done and the review load matches the number of submissions.
- [ ] Single PDF: paper + supplementary after the references.
- [ ] Upload code as supplementary (encouraged), anonymised per step 3.
- [ ] Submit ≥ 24 h early. AOE is UTC−12; a "Sep 25" that feels like Friday evening locally is
      not the deadline you think it is.

---

## Standing rule

Every number in the paper traces to a `experiments/runs/` directory, and every claim traces to
a RESULTS.md section that is marked **current**. This project has already published three
claims that a proper baseline then refuted; the ledger is what stops a fourth.
