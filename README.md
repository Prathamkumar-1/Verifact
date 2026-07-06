# Verifact  — A Multi-Agent Claim Verification System

> **Don't just ask an LLM whether something is true — make it gather evidence first.**
>
> Verifact is a multi-agent system that takes a claim or news headline and returns
> a labelled verdict (**True / False / Mixed / Unverified**) with a confidence score,
> plain-language reasoning, and source citations. Built with LangGraph, LangChain,
> Groq-hosted Llama models, and a small RAG layer over gathered evidence.

*Capstone project — Agentic AI, Learners' Space 2026 (Week 4).*

---

## Why this problem?

Misinformation spreads faster than corrections can keep up with, and a single
LLM asked "is this true?" will happily hallucinate a confident answer. Real
fact-checking is a *workflow*: you decompose the claim, hunt for evidence from
independent sources, weigh how trustworthy those sources are, and only then
reach a verdict. Each of those steps needs different tools and different
reasoning — which is exactly the case for **multiple cooperating agents** rather
than one big prompt.

That's Verifact's whole pitch: five specialised agents, each with a clear job,
coordinated by a supervisor, with evidence gathered in parallel and read through
a retrieval step before anyone is allowed to judge.

---

## The five agents (+ a human-in-the-loop gate)

| # | Agent | Responsibility | Why it can't be merged with the others |
|---|-------|----------------|----------------------------------------|
| 1 | **Planner** | Breaks the claim into 2–4 atomic, web-searchable sub-questions. | Decomposition is a distinct skill from judging; getting this right determines what the researchers even look for. |
| 2 | **Researcher** (×N, in parallel) | For each sub-question: runs web search + Wikipedia, then distils raw results into tidy `Evidence` rows. | Runs once per sub-question concurrently; uses the fast/cheap model and a tool-calling loop. |
| 3 | **Evidence Analyst** | Pulls the most relevant chunks via **RAG** (HF embeddings + FAISS) and sorts them into supporting / refuting / open. | Needs retrieval, not search — it reasons over *already-gathered* evidence. |
| 4 | **Credibility Analyst** | Scores source quality, recency, and cross-source agreement; flags bias/contradiction. | A separate lens from content analysis: two weak blogs agreeing is very different from two Reuters reports agreeing. |
| 5 | **Judge** | Combines the analyst's summary + the credibility report into the final structured `Verdict`. | The only agent allowed to commit to a label, and it's deliberately downstream of all the others so it can't shortcut. |
| 6 | **Approval Gate** *(HITL)* | Pauses the graph after the Judge rules so a human can approve the verdict, or reject it and send it back with feedback. | High-stakes decisions shouldn't be finalized silently — the Week 4 notes call this out explicitly. |

A seventh role, the **Supervisor**, isn't a "worker" — it's the coordinator that
decides which agent runs next.

---

## Orchestration

Verifact deliberately combines **three** of the patterns:

### 1. Supervisor (primary pattern)
A coordinator node inspects the shared state and routes to the next agent
dynamically — there is **no fixed pipeline**. Critically, the supervisor can
send the system *back for another research round* if too little evidence was
gathered, up to a cap. 

### 2. Parallel + Aggregator (map-reduce)
The research step is a fan-out: one `Send("researcher", {...})` is spawned per
sub-question, all running in the same super-step. Their returned evidence lists
are **aggregated** by an `operator.add` reducer on the shared state — the
classic map-reduce shape. The two analysts likewise run in parallel.

### 3. Human-in-the-loop checkpoint (failure handling)
After the Judge rules, the **Approval Gate** node calls LangGraph's
`interrupt()` to pause the graph and surface the proposed verdict to a human.
The gate returns a `Command(goto=..., update=...)` — Week 4's "route + update in
one step" pattern — to either retry the Judge with the reviewer's feedback or
finalize. The CLI driver detects the pause via `get_state(config).interrupts`,
prompts the user, and resumes with `Command(resume=HumanReview(...))`.

### The graph

```
                         START
                           │
                           ▼
                    ┌─► supervisor ◄──────────────────────────┐
                    │    (routes by state)                     │
                    │   ┌─────┬───────────┬─────────┐           │
                    │   ▼     ▼           ▼         ▼           │
                    │ planner  start_research   analyze_step   judge
                    │            │  (×N)            │  (×2)       │
                    │            ▼                 ▼             ▼
                    │        researcher ──┐  evidence_analyst   approval_gate
                    │                     │  credibility_        (HITL)
                    │                     │    analyst             │
                    │                     │                        ▼
                    └─────────────────────┴────────────────────► finalize ──► END
                                                              (Command routes
                                                               approve / retry)
```

- `supervisor` → conditional edges to `planner` / `start_research` / `analyze_step` / `judge`
- `start_research` → fans out into N parallel `researcher` tasks via `Send(...)`
- `analyze_step` → both analysts run concurrently, then converge back at the supervisor
- `judge` → `approval_gate`, which routes via `Command` to `judge` (retry) or `finalize`

### State management
we use a **shared global state**.

| State key | Who writes it | Notes |
|---|---|---|
| `claim` | caller (input only) | set once, never overwritten |
| `sub_questions` | Planner | |
| `evidence` | Researcher (×N) | `operator.add` reducer — aggregates parallel writes |
| `evidence_summary` | Evidence Analyst | |
| `credibility` | Credibility Analyst | |
| `verdict` | Judge | |
| `judge_feedback` | Approval Gate | fed back to the Judge on rejection |
| `approved` / `hitl_rejections` | Approval Gate / Finalize | HITL bookkeeping |
| `next`, `supervisor_note` | Supervisor | routing decisions |
| `research_rounds` | `start_research` | caps the re-research loop |

The shared-state choice was confirmed acceptable for supervisor patterns of this
size (per instructor guidance in the cohort channel).

---

## How a run flows (step by step)

1. **You submit a claim** via the CLI.
2. The **Supervisor** sees no plan yet → routes to the **Planner**.
3. The **Planner** returns 2–4 sub-questions.
4. **Supervisor** → `start_research`, which **fans out** one **Researcher** per
   sub-question. Each researcher hits web search (Tavily, or keyless DuckDuckGo)
   and Wikipedia, then distils results into `Evidence` rows. Results aggregate.
5. If the evidence pile is still too thin, the **Supervisor** loops back for
   another research round (capped at `MAX_RESEARCH_ROUNDS`).
6. **Supervisor** → `analyze_step`: the **Evidence Analyst** (RAG-backed) and
   the **Credibility Analyst** run **in parallel**, each writing its findings.
7. **Supervisor** → **Judge**, which issues a proposed structured `Verdict`.
   *(If the verdict is malformed, the Judge retries with a corrective prompt;
   if all retries fail, it emits a conservative fallback — see below.)*
8. **Approval Gate** pauses the graph (`interrupt()`). The CLI shows you the
   proposed verdict and asks: accept, or reject + give feedback. On rejection
   the Judge reruns with your note; otherwise the verdict is finalized.
9. The CLI pretty-prints the final verdict, the evidence summary, the
   credibility scores, and the citations.

---

## Failure handling

| Mechanism | Where | What it does |
|---|---|---|
| **1. Retries** | `judge_agent` | If the Judge returns an empty/malformed verdict (or the API errors), it retries up to `JUDGE_MAX_RETRIES` times, each retry appending a *"your previous output was invalid, fix it"* instruction so the model self-corrects. |
| **2. Fallback agent** | `_judge_fallback` | If every retry fails, the Judge emits a conservative `unverified` verdict with confidence 0 instead of raising — the run completes and the human reviewer sees that the system couldn't decide. |
| **3. Human-in-the-loop** | `approval_gate` | The headline mechanism: after the Judge rules, the graph **pauses** and surfaces the verdict for human approval. The reviewer can accept it or reject it with feedback (which routes back to the Judge). Implemented with LangGraph `interrupt()` + `Command(resume=...)`. |

Every external call (web search, Wikipedia, RAG) is additionally wrapped so a
flaky source degrades the result rather than crashing the run.

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | **LangGraph** | `StateGraph`, `Send` (parallel fan-out), `interrupt()` + `Command(resume/goto)` for HITL, `InMemorySaver` checkpointer |
| Agent framework | **LangChain** | `Runnable`, `with_structured_output`, tool wrappers |
| LLMs | **Groq** — `llama-3.3-70b-versatile` (reasoning agents) + `llama-3.1-8b-instant` (researchers) | Fast and free-tier friendly |
| Web search | **Tavily** (if keyed) → **DuckDuckGo** fallback | Graceful degradation to keyless |
| Reference | **Wikipedia** | Keyless, adds a stable second source |
| RAG | **HuggingFace embeddings** (`all-MiniLM-L6-v2`) + **FAISS** | Local, no API key; degrades to raw snippets if unavailable |
| Output | **Pydantic v2** structured outputs | Every agent returns a validated object |
| CLI | **argparse** + **rich** | Pretty verdict panel; plain-text fallback |

---

## Setup

You need **Python 3.10+** and a free **Groq** API key. That's the only hard
requirement — everything else is optional and degrades gracefully.

```bash
# 1. Clone & enter
git clone <your-repo-url> verifact
cd verifact

# 2. (Recommended) virtual environment
python -m venv .venv
#   Windows:  .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your key
cp .env.example .env          # Windows: copy .env.example .env
#   then edit .env and paste your GROQ_API_KEY from https://console.groq.com/keys
```

**Keys at a glance**

| Variable | Required? | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | https://console.groq.com/keys (free) |
| `TAVILY_API_KEY` | Optional (recommended) | https://tavily.com — free 1000 calls/mo; improves search reliability. Without it, Verifact uses keyless DuckDuckGo. |

A few optional **runtime knobs** are also read from the environment (see
`.env.example`): `VERIFACT_HITL` (on/off for the approval gate),
`VERIFACT_JUDGE_RETRIES`, `VERIFACT_HITL_MAX_REJECTIONS`, and
`VERIFACT_MAX_RESEARCH_ROUNDS`.

No key is needed for Wikipedia or the local RAG model.

---

## Usage

```bash
# Verify any claim (with human approval gate — the default)
python run.py "The Great Wall of China is visible from space."

# Run a built-in sample claim (1–6)
python run.py --example 2
python run.py --example 5

# List the sample claims
python run.py --list

# Watch each agent step stream by
python run.py --verbose "5G networks spread COVID-19."

# Fully automated run — skip the human approval gate (good for batch tests)
python run.py --no-hitl "Eating eggs is bad for your health."

# Run the gate but auto-approve without prompting
python run.py --yes "An AI beat the world champion at Go in 2016."
```

When the human-in-the-loop gate is on, you'll see a prompt like this once the
Judge has produced a verdict:

```
============================================================
HUMAN APPROVAL REQUIRED (Week 4 HITL checkpoint)
============================================================
  Proposed verdict: FALSE   (confidence 88%)
  Summary: The Great Wall is not visible from the unaided eye in space.
  Reasoning: Multiple astronauts and NASA sources confirm...
============================================================
  [a]ccept   [r]eject (let the Judge retry with your feedback)
  Your choice [a/r]:
```

### Example output

```
Verifying: "An AI program defeated the human world champion at the game of Go in 2016."
────────────────────────────────────────────────────────────────────────────────────

┌ VERDICT ───────────────────────────────┐
│ TRUE   confidence: 95%                  │
└─────────────────────────────────────────┘

Claim:   An AI program defeated the human world champion at the game of Go in 2016.
Summary: AlphaGo, developed by DeepMind, beat Lee Sedol 4–1 in March 2016.

Reasoning:
Multiple reputable sources confirm the March 2016 AlphaGo vs. Lee Sedol match...
...
```

*(The exact wording is generated live by the model.)*

---

## Project structure

```
verifact/
├── __init__.py
├── config.py       # env vars, model names, failure-handling knobs
├── schemas.py      # Pydantic models: Evidence, ResearchPlan, Verdict, HumanReview, ...
├── tools.py        # web search (Tavily→DDG), Wikipedia, RAG retriever
├── agents.py       # the 5 agents + approval gate + supervisor router
└── graph.py        # VerifactState, build_graph() — the orchestration
run.py              # CLI entrypoint (drives the HITL pause/resume loop)
examples.py         # 6 sample claims for demos
tests/
└── test_smoke.py   # offline tests (no API key needed)
requirements.txt
.env.example
README.md
```

## Running the tests

```bash
python tests/test_smoke.py        # or: python -m pytest tests/ -v
```

These are offline smoke tests: they confirm the schemas validate, the graph
compiles, and the evidence reducer aggregates correctly. No API key required.

---

## References

- [LangGraph — multi-agent systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph — map-reduce / parallelism (`Send`)](https://langchain-ai.github.io/langgraph/how-tos/map_reduce/)
- [LangGraph — human-in-the-loop (`interrupt` + `Command(resume=...)`)](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph — `Command` (routing + state update)](https://reference.langchain.com/python/langgraph/types/Command)
- [LangChain — structured outputs](https://python.langchain.com/docs/how_to/structured_output/)
- [ChatGroq integration](https://python.langchain.com/docs/integrations/chat/groq/)
- [GroqCloud supported models](https://console.groq.com/docs/models)

---

## License

MIT — see the repo. Built as a capstone for the Agentic AI, Learners' Space 2026 bootcamp.
