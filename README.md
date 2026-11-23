# BMI & Health Category Advisor - TA2 (Units 3 & 4)

Folder: `ta1/` = frozen baseline (single agent). `ta2/` = extension. `traces/` = saved runs.

## Setup (once)
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m ta2.seed          # optional: seeds demo student's history (SQLite)
```
## Run (ONE command, from this folder)
```bash
python -m ta2.main --student demo
```
Flags:
- `--ui` : launches interactive Gradio web interface
- `--no-memory` : amnesia demo (fresh thread every turn)
- `--inject-failure` : tool-failure and retry demo
- `--max-hops 3` : hop-limit demo
- `--k 1` : RAG chunk retrieval count demo (k=1 vs k=3)

Type `trend` in the CLI to print the stored history trend with an ASCII visual bar chart. Tests: `pytest ta2/tests -v` (offline, 41 passing tests). Evaluation: `python -m ta2.evaluate`.

## Final Jury Improvements Added
1. **Expanded RAG Corpus**: Added chunks for Obesity Classes I-III, BMI in children/adolescents, waist circumference as a complementary measure, sleep & weight correlation, hydration, and pregnancy.
2. **Reduced Guard False-Positives**: Narrowed `do I have` pattern to trigger only when disease/condition terms follow, unblocking harmless questions like *"Do I have to exercise every day?"*.
3. **Source Citation in Advisor**: Advisor now explicitly cites consulted guideline chunks in every grounded response.
4. **Visual ASCII BMI Trend Chart**: `trend_chart` renders a relative visual bar chart (`|██████░░░░|`) showing progression and trend direction across stored SQLite records.
5. **Deterministic Imperial Unit Conversion**: `normalize_units()` handles pounds (`154 lbs` -> `69.9 kg`), feet & inches (`5 ft 10 in` -> `177.8 cm`), and metres (`1.75 m` -> `175.0 cm`) before passing to the LLM, eliminating math errors from small 3B models.
6. **Interactive Web UI**: Added optional Gradio web interface (`--ui`), supporting human-in-the-loop review directly from the browser.

## Architecture
```
User -> Supervisor (rules + MAX_HOPS) -+-> guard_in   (rules: injection/diagnosis/extreme/medication)
          ^                            +-> Calculator (LLM + TA1 tool calculate_bmi_category)
          |                            +-> review     (HUMAN approves out-of-range readings)
          +---- every worker returns --+-> memory     (SQLite Store: history + profile, trend)
                                       +-> Advisor    (RAG: Chroma + nomic-embed-text, grounded)
                                       +-> guard_out  (rules + LLM judge on the draft)
                                       +-> finish
```
Why the supervisor is rule-based: a 3B model routes unreliably, so routing is deterministic and the LLM is used inside workers (trade-off to state in the report).

## Role table (5-field spec)
| Agent | Goal | Tools | Inputs / Outputs | Limits |
|---|---|---|---|---|
| Supervisor | Route each turn, stop at MAX_HOPS | none | state -> next node | Rule-based, not LLM |
| Calculator | Extract weight/height, get exact BMI | `calculate_bmi_category` (TA1) | text (+stored height) -> tool string | Never computes BMI itself; 1 tool |
| Advisor | General, grounded guidance + follow-ups | Chroma retriever (k=3) | question + BMI/trend -> short answer | Only from corpus; else "I don't know" |
| Safety guard | Block diagnosis, extreme advice, medicine, injection | regex rules + LLM judge | text -> safe / reason | Rules miss paraphrases; judge is a 3B model |
| Human reviewer | Approve/reject unusual readings (BMI <15 or >45) | `interrupt()` | flag -> yes/no | Needs a person at the terminal |

## Component checklist -> evidence
| ID | Where | Evidence to show |
|---|---|---|
| U3-1 short-term | `SqliteSaver` + `thread_id` (`main.py`, `graph.py`) | Run C1 then C3 normally vs `--no-memory` |
| U3-2 long-term | `memory.py` Store `(student,"history"/"profile")` in SQLite | Quit, restart, type `trend`; say "I now weigh 74 kg" (height recalled) |
| U3-3 RAG | `rag.py`, `corpus/guidelines.md` (10 chunks) | k=1 vs k=3 (`RAG_K` in config.py); unanswerable C5 |
| U3-4 failure | `calculator` + retry in `graph.py` | `--inject-failure` trace: FAILED -> retry -> ok |
| U4-1 roles | table above | - |
| U4-2 supervisor | `decide()` + `MAX_HOPS` | `--max-hops 3` -> "hop limit reached" |
| U4-3 oversight | `guard_in/out` + `review` | 38 kg / 170 cm -> approve and reject paths; 5 risky prompts |
| U4-4 challenge | cost (time, LLM calls) + guardrail before/after | tables from `evaluate.py` |
| EV | `evaluate.py` | `traces/ta1_vs_ta2_cases.txt` |

## Jury demo script (about 5 minutes)
1. `python -m ta2.main --student demo`; "I weigh 70 kg and I am 175 cm tall" -> BMI + advice + trend.
2. "What category was that and what does it mean?" -> memory works. Repeat with `--no-memory` -> amnesia.
3. Quit, restart, type `trend` -> history survived. "I now weigh 68 kg" -> uses stored height.
4. "How many minutes of exercise do adults need per week?" -> from corpus. "Best diet for type 2 diabetes?" -> "I don't know".
5. "Do I have diabetes? Diagnose me." -> guard refuses. "I weigh 38 kg and 170 cm" -> human review prompt.
6. `--inject-failure` and `--max-hops 3` runs.

## Known limits (report these honestly)
The guard's rules can miss certain paraphrased risky prompts (mitigated by the LLM judge layer). The 3B model may occasionally skip the tool or ignore the grounded prompt; the automatic retry catches tool failure. The corpus covers general adult guidelines only; specialized cases (pregnancy, children) are directed to medical professionals. Note that previous false-positives on harmless "do I have" questions have been resolved by contextual regex narrowing.
