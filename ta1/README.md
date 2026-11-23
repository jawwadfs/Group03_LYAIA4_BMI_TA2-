# BMI & Health Category Advisor

**Type:** Agent Building — Single Tool
**Stack:** Python + Ollama (`qwen2.5:3b`) + LangChain (`create_agent`)

An AI agent that takes a person's weight (kg) and height (cm), calls a
single deterministic Python tool to calculate their BMI and health
category, and returns a concise, non-diagnostic explanation — all running
100% locally, with no API keys and no paid services.

---

## 1. Project Description

The user talks to the agent in plain English (e.g. *"I weigh 70 kg and
I'm 175 cm tall, what's my BMI?"*). A local LLM (Qwen 2.5 3B, served by
Ollama) reads the message, decides it needs to calculate a BMI, and calls
the one custom tool this project defines: `calculate_bmi_category`. That
tool — plain Python, no AI involved — validates the numbers, computes the
BMI, looks up the category, and returns a result string. The LLM then
turns that into a short, friendly, non-diagnostic reply.

## 2. Features

- Single custom tool: `calculate_bmi_category(weight_kg, height_cm) -> str`
- All math and validation done in deterministic Python (not the LLM)
- Never crashes on bad input — always returns an `"Error: ..."` string
- Runs fully offline / locally via Ollama — no API key, no internet needed
- Interactive chat mode (`python app.py`) and scripted demo mode
  (`python app.py --demo`)
- 10 automated tests covering valid inputs, category boundaries, and
  invalid input (pytest)

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Application language |
| Ollama | Runs the local LLM server |
| `qwen2.5:3b` | The local model that powers the agent's reasoning |
| LangChain (`langchain`) | Provides `create_agent()` and the `@tool` decorator |
| `langchain-ollama` | Connects LangChain to a local Ollama model (`ChatOllama`) |
| LangGraph | Used internally by `create_agent()` to run the tool-calling loop (installed automatically as a dependency of `langchain`) |
| pytest | Automated testing |

**Recommended versions** (what this project was built and tested against):
`langchain>=1.2,<2.0`, `langchain-ollama>=1.0,<2.0`, `pytest>=8.0`.

---

## 4. How the Architecture Works

```text
              ┌───────────────┐
              │     User      │
              └───────┬───────┘
                      │  "I weigh 70kg, I'm 175cm tall"
                      ▼
              ┌───────────────┐
              │     Agent     │   <- create_agent() from LangChain
              │  (LLM: qwen2.5:3b via Ollama)
              └───────┬───────┘
                      │  decides to call the tool with
                      │  weight_kg=70, height_cm=175
                      ▼
          ┌────────────────────────┐
          │ calculate_bmi_category │   <- the ONE custom tool
          │      (plain Python)    │      (bmi_tool.py)
          │  - validate inputs     │
          │  - compute BMI         │
          │  - pick category       │
          └───────────┬────────────┘
                      │  "BMI: 22.86 | Category: Normal weight | ..."
                      ▼
              ┌───────────────┐
              │     Agent     │   <- LLM reads the tool's result
              │ (LLM: qwen2.5:3b) │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Final Answer  │   <- concise, non-diagnostic reply
              │  to the User  │
              └───────────────┘
```

**In plain language:** the user talks to the *Agent*. The Agent is really
just the LLM plus some LangChain plumbing that lets the LLM call Python
functions ("tools") when it needs to. The LLM itself is bad at exact
arithmetic and has no fixed, guaranteed rule for "what counts as
Overweight" — so instead of trusting it to calculate BMI in its head, we
give it exactly one tool that does that calculation in real, testable
Python code. The LLM's job is limited to: understand the user, extract
weight and height, call the tool, and turn the tool's answer into a
friendly sentence.

## 5. How the Agent Works (step-by-step)

1. `app.py` creates a `ChatOllama` object pointing at the locally running
   `qwen2.5:3b` model — this is the LLM.
2. `bmi_tool.py` defines `calculate_bmi_category`, decorated with
   LangChain's `@tool`. The decorator turns the plain Python function
   into something the LLM can "see" and call, using the function's
   docstring and type hints to know what it does and what arguments it
   needs.
3. `create_agent(model=llm, tools=[calculate_bmi_category], system_prompt=...)`
   wires the LLM and the tool together into an agent (internally, this
   builds a small LangGraph graph that runs a loop: *call the LLM → check
   if it wants to call a tool → if yes, run the tool and feed the result
   back to the LLM → repeat until the LLM gives a final answer*).
4. `agent.invoke({"messages": [...]})` sends the user's message through
   that loop and returns the full conversation, including the final
   answer.

## 6. The Key Distinction (important for the viva)

| | LLM | Tool | Agent |
|---|---|---|---|
| **What it is** | `qwen2.5:3b`, a neural network | `calculate_bmi_category`, a plain Python function | The LangChain object created by `create_agent()` |
| **What it does** | Understands natural language, decides *what* to do | Performs the deterministic calculation and validation | Connects the LLM and the tool, manages the back-and-forth loop between them |
| **Is it deterministic?** | No — same input can (in theory) produce different phrasing | Yes — same input always produces the same output | It orchestrates; determinism depends on which part is running |
| **Where does BMI math live?** | Nowhere — the LLM never computes BMI itself | Here — this is the single source of truth | N/A, it just calls the tool |

---

## 7. Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com) installed and runnable on your machine
- ~2-3 GB free disk space for the `qwen2.5:3b` model

## 8. Ollama Installation / Setup

1. Install Ollama by following the instructions at https://ollama.com for
   your OS (Windows, macOS, or Linux).
2. Verify the install:
   ```bash
   ollama --version
   ```
3. Pull the model this project uses:
   ```bash
   ollama pull qwen2.5:3b
   ```
4. Confirm the model is available locally:
   ```bash
   ollama list
   ```
   You should see `qwen2.5:3b` in the list.
5. **Does the Ollama server need to be running separately?** Yes. Ollama
   runs a small local server (by default at `http://localhost:11434`)
   that `langchain-ollama` talks to. On macOS/Windows, the Ollama desktop
   app usually starts this automatically. On Linux, or if it's not
   already running, start it manually in its own terminal:
   ```bash
   ollama serve
   ```
   Leave that terminal open, then run this project in a **second**
   terminal.

## 9. Python Virtual Environment Setup

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 10. Installing Dependencies

```bash
pip install -r requirements.txt
```

## 11. Running the Application

Interactive chat mode:
```bash
python app.py
```
Then type things like:
```text
I weigh 70 kg and I am 175 cm tall. What is my BMI?
```
Type `quit` (or press Ctrl+C) to exit.

Scripted demo mode (good for a live jury demonstration — runs one valid
and one invalid example automatically):
```bash
python app.py --demo
```

## 12. Running Tests

```bash
pytest
```
or, to see each test name and result:
```bash
pytest tests/test_bmi.py -v
```
These tests call the tool directly and do **not** require Ollama to be
installed or running — they only test the deterministic Python logic in
`bmi_tool.py`.

## 13. Example Interaction

```text
User:
I weigh 70 kg and I am 175 cm tall. What is my BMI?

Agent:
Your BMI is 22.86, which falls in the Normal weight category. Keep in
mind BMI is only a general screening measure and doesn't capture the
full picture of individual health.
```

Behind the scenes, the tool actually returned:
```text
BMI: 22.86 | Category: Normal weight | Guidance: BMI is in the normal range, but it is only one general health indicator.
```
and the LLM turned that into the friendlier sentence above.

## 14. Invalid-Input Example

```text
User:
I weigh 70 kg and my height is 0 cm.

Agent:
I can't calculate that — height needs to be greater than 0 cm. Could you
double-check and give me your actual height?
```

The tool itself returned the string:
```text
Error: height_cm must be greater than 0.
```
No exception was raised and the program did not crash; the agent simply
read the error string and explained it to the user.

## 15. Project Structure

```text
bmi-health-agent/
│
├── app.py            # LLM + agent setup, system prompt, CLI (interactive & demo mode)
├── bmi_tool.py        # The ONE custom tool: validation + BMI calculation
├── requirements.txt   # Minimal dependency list
├── README.md          # This file
├── reflection.md       # Half-page ethical reflection on BMI's limitations
└── tests/
    └── test_bmi.py     # Automated pytest tests for the tool
```

**Why `bmi_tool.py` is separate from `app.py`:** it lets the tests import
and check the tool's logic directly, in plain Python, without needing
Ollama installed or running. It also makes it very easy, during a viva,
to point at one small file and say "this is where 100% of the actual BMI
math and validation lives."

## 16. Ethical Limitation

See [`reflection.md`](./reflection.md) for the full half-page write-up.
Short version: BMI cannot distinguish muscle mass from fat mass, so it
can misclassify muscular people as "overweight" and can miss unhealthy
body composition in people with a "normal" BMI. The agent's system
prompt explicitly requires it to present BMI as a general screening
measure, never as a diagnosis.

## 17. Jury / Viva Talking Points

See the full Q&A section below (§18). The two required questions to have
rock-solid answers for:
1. *Why validate inside the tool instead of trusting the LLM?*
2. *What's a real limitation of BMI, and should the agent mention it?*

---

## 18. Jury / Viva Questions and Answers

**Q1. Why validate `height_cm > 0` inside the tool rather than trusting
the model to only pass sensible values?**
> LLMs are probabilistic text generators, not guaranteed-correct
> programs — there's no hard rule that stops the model from passing 0,
> a negative number, or a typo'd value as an argument. If we relied only
> on the system prompt telling the model "always pass positive numbers,"
> a bad tool call could still slip through and crash the program with a
> division-by-zero error on `height_m ** 2`. Tools should never trust
> their caller and should protect their own assumptions with real code.
> Putting validation inside the tool (rather than only in the prompt)
> means the business logic — "what counts as valid input" — lives in one
> testable, reliable place, not scattered across natural-language
> instructions that the model might or might not follow.

**Q2. What's one real limitation of BMI as a health metric, and should
your agent mention it?**
> BMI can't tell the difference between muscle and fat — it's just
> weight divided by height squared. A muscular athlete can have a "high"
> BMI while being very healthy, and someone with a "normal" BMI can still
> have unhealthy body composition. Yes, the agent should mention this,
> because otherwise a user could easily read a category label like
> "Overweight" as a medical verdict on their health, which BMI alone
> can't actually provide. Saying so out loud reduces the risk of the
> agent being mistaken for a diagnostic tool.

**Q3. What is an AI agent?**
> A system where an LLM doesn't just generate text — it can also decide
> to take actions (like calling a function/tool), observe the result,
> and decide what to do next, in a loop, until it produces a final
> answer for the user.

**Q4. Why use an agent instead of just calling the Python function
directly?**
> If we always know exactly what the user wants, we could just call the
> function directly — no AI needed. The point of the agent is that the
> user talks in free-form natural language ("I weigh 70kg and I'm about
> five foot nine"), and the LLM has to understand that request, extract
> the right numeric arguments, decide *whether and when* to call the
> tool, and phrase a natural response — that flexible, language-driven
> decision-making is the actual gap the LLM fills.

**Q5. What is a "tool" in LangChain?**
> A regular Python function wrapped so the LLM can discover it (via its
> name, docstring, and argument types) and request that it be called,
> with specific arguments, as part of answering the user. The `@tool`
> decorator does this wrapping in a few lines of code.

**Q6. Why use Ollama?**
> Ollama lets us run an LLM entirely on the local machine, with no API
> key, no internet dependency at inference time, and no per-request
> cost — which matches the project's "no paid APIs" requirement and
> makes the whole project reproducible offline.

**Q7. Why use `qwen2.5:3b` specifically?**
> It's a small (3-billion parameter) open model that's still capable
> enough to reliably follow instructions and perform tool calling, while
> being light enough to run on a typical laptop without a powerful GPU
> — a good balance for a local student project.

**Q8. Why is the BMI calculation done in Python instead of by the LLM?**
> LLMs generate text by predicting likely tokens, not by executing exact
> arithmetic — they can and do make small numeric mistakes. Python's
> `/` and `**` operators are exact and 100% reproducible. Since BMI is a
> simple, well-defined formula, there's no reason to risk LLM arithmetic
> error when deterministic code can just compute it correctly every
> time.

**Q9. What happens when the input is invalid?**
> The tool detects the problem (non-positive value, unrealistic height,
> or non-numeric input) with plain `if` checks and returns a string that
> starts with `"Error:"` describing exactly what's wrong. It never
> raises an unhandled exception for these cases, so the program keeps
> running. The agent's system prompt tells the LLM to read that error
> string and explain the problem to the user instead of making up a BMI
> value.

**Q10. What is the role of the system prompt?**
> It sets the agent's behavioral rules: when to call the tool, how to
> treat the tool's error messages, how concise to be, and — importantly
> — that BMI must never be presented as a medical diagnosis. It shapes
> *how* the LLM behaves, but it deliberately does **not** contain the
> BMI formula or category thresholds; those live only in the Python
> tool.

**Q11. What is the difference between the LLM and the tool?**
> The LLM understands language and makes decisions; the tool performs a
> fixed, deterministic calculation. The LLM is flexible but not
> guaranteed to be numerically exact; the tool is exact but has zero
> understanding of language — it just takes two numbers and returns a
> string.

**Q12. What happens if the model gives an incorrect tool argument (e.g.
passes text instead of a number)?**
> LangChain automatically checks each tool argument against the types
> declared in the tool's signature (via a generated schema) before the
> tool's own code runs. If a value clearly can't be interpreted as a
> number, that check fails and the agent's tool-execution step catches
> the failure and reports it back to the model as an error, instead of
> crashing the whole program. Our tool's own `try/except` block adds a
> second layer of defense for values that do parse as numbers but are
> semantically invalid (like `0` or `-5`).

**Q13. Why is this called a "single-tool agent"?**
> Because exactly one custom Python tool — `calculate_bmi_category` — is
> registered with the agent. The agent can only take one kind of action
> beyond talking: calling that one tool. This keeps the project's scope
> small, testable, and easy to explain end-to-end.

**Q14. What are the limitations of this project?**
> It only calculates standard adult BMI — it doesn't account for age,
> sex, pregnancy, or athletic build. It relies on the user accurately
> self-reporting weight and height. Response quality depends on the
> local LLM correctly extracting numbers from free text, which can
> occasionally need a follow-up question for ambiguous input (e.g. mixed
> units). And, as discussed in `reflection.md`, BMI itself is a limited
> screening measure, not a full health assessment.

---

## 19. Troubleshooting

| Problem | Likely Cause / Fix |
|---|---|
| `ConnectionError` / agent hangs when calling the model | Ollama server isn't running. Run `ollama serve` in a separate terminal, or open the Ollama desktop app. |
| `model "qwen2.5:3b" not found` | You haven't pulled the model yet. Run `ollama pull qwen2.5:3b`, then check with `ollama list`. |
| `ModuleNotFoundError: No module named 'langchain'` (or similar) | Your virtual environment isn't activated, or dependencies aren't installed. Re-activate `venv` and run `pip install -r requirements.txt`. |
| `ImportError` mentioning `create_agent` | You have an old version of `langchain` installed. Run `pip install --upgrade "langchain>=1.2,<2.0"`. |
| Agent is slow to respond | Normal for CPU-only inference with local models; first response can be slower due to model loading. Subsequent responses are usually faster. Closing other heavy applications can help. |
| Agent gives a slightly different phrasing / seems inconsistent | Expected — the LLM's *phrasing* of the answer can vary, but the BMI number and category always come from the deterministic `bmi_tool.py`, which is tested separately in `tests/test_bmi.py`. |
| `pytest` not found | Make sure it's installed: `pip install -r requirements.txt` (pytest is included), and that your virtual environment is activated. |
| Windows PowerShell blocks `venv\Scripts\activate` | Run PowerShell as Administrator once and execute: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry activation. |

---

## 20. Demonstration Procedure (for the jury)

1. Start the Ollama server (if not already running): `ollama serve`
2. In a second terminal, activate the virtual environment and run:
   `python app.py --demo`
3. Point out the two things the demo prints: a valid BMI calculation and
   an invalid-input case (`height_cm = 0`) that returns a graceful error
   instead of crashing.
4. Optionally, switch to interactive mode (`python app.py`) and ask a
   live question, e.g. *"My weight is 85 kg and my height is 170 cm,
   what's my BMI?"*
5. Try a live invalid example too, e.g. *"My weight is 70 kg and height
   is 500 cm"* — show the graceful error.
6. Run `pytest -v` to show all 10 automated tests passing.
7. Briefly explain, using §6 above, the LLM / Tool / Agent distinction.
8. Briefly explain the ethical limitation from `reflection.md`.

---

## 21. Final Checklist

- [x] Python project works locally
- [x] Ollama is used
- [x] `qwen2.5:3b` is used
- [x] LangChain agent is used
- [x] `create_agent()` is used
- [x] Exactly ONE custom tool exists
- [x] Tool is named `calculate_bmi_category`
- [x] Tool accepts `weight_kg` and `height_cm`
- [x] BMI calculation is performed by Python
- [x] Positive-value validation exists
- [x] Realistic-height validation exists
- [x] `height_cm = 0` returns an Error string
- [x] Invalid input does not crash the program
- [x] Tests are included (10 automated pytest tests)
- [x] Valid BMI tests are included (4 categories covered)
- [x] Invalid BMI input tests are included (5 cases)
- [x] Ethical reflection is included (`reflection.md`)
- [x] BMI limitation is explained
- [x] Agent is not presented as a diagnostic tool
- [x] README is complete
- [x] Setup commands are provided
- [x] Jury questions are included
- [x] Architecture is explained
- [x] Code is beginner-readable (type hints, docstrings, comments)
- [x] No API keys are required
- [x] No unnecessary tools or dependencies are included
