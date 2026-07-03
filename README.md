# Verifact 🔍 — A Multi-Agent Claim Verification System

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

## The five agents

| # | Agent | Responsibility | Why it can't be merged with the others |
|---|-------|----------------|----------------------------------------|
| 1 | **Planner** | Breaks the claim into 2–4 atomic, web-searchable sub-questions. | Decomposition is a distinct skill from judging; getting this right determines what the researchers even look for. |
| 2 | **Researcher** (×N, in parallel) | For each sub-question: runs web search + Wikipedia, then distils raw results into tidy `Evidence` rows. | Runs once per sub-question concurrently; uses the fast/cheap model and a tool-calling loop. |
| 3 | **Evidence Analyst** | Pulls the most relevant chunks via **RAG** (HF embeddings + FAISS) and sorts them into supporting / refuting / open. | Needs retrieval, not search — it reasons over *already-gathered* evidence. |
| 4 | **Credibility Analyst** | Scores source quality, recency, and cross-source agreement; flags bias/contradiction. | A separate lens from content analysis: two weak blogs agreeing is very different from two Reuters reports agreeing. |
| 5 | **Judge** | Combines the analyst's summary + the credibility report into the final structured `Verdict`. | The only agent allowed to commit to a label, and it's deliberately downstream of all the others so it can't shortcut. |

A sixth role, the **Supervisor**, isn't a "worker" — it's the coordinator that
decides which agent runs next.

---

## Orchestration

Verifact uses **two** of the four orchestration patterns from the course, on purpose:

### 1. Supervisor (primary)
A coordinator node inspects the shared state and routes to the next agent
dynamically — there is **no fixed pipeline**. Critically, the supervisor can
send the system *back for another research round* if too little evidence was
gathered, up to a cap. That dynamic re-routing is what distinguishes a real
supervisor from a glorified linear chain.

### 2. Parallel + Aggregator (map-reduce)
The research step is a fan-out: one `Send("researcher", {...})` is spawned per
sub-question, all running in the same super-step. Their returned evidence lists
are **aggregated** by an `operator.add` reducer on the shared state — the
classic map-reduce shape. The two analysts likewise run in parallel.

### The graph

```
                         START
                           │
                           ▼
                    ┌─► supervisor ◄────────────────────┐
                    │    (routes by state)               │
                    │   ┌─────┬───────────┬─────────┐     │
                    │   ▼     ▼           ▼         ▼     │
                    │ planner  start_research   analyze_step   judge ──► END
                    │            │  (×N)            │  (×2)
                    │            ▼                  ▼
                    │        researcher ──┐    evidence_analyst ──┐
                    │                     │    credibility_analyst┘
                    └─────────────────────┴───────────────────────┘
```

- `supervisor` → conditional edges to `planner` / `start_research` / `analyze_step` / `judge`
- `start_research` → fans out into N parallel `researcher` tasks via `Send(...)`
- `analyze_step` → both analysts run concurrently, then converge back at the supervisor
- only `judge` reaches `END`

The shared state (`VerifactState`) is a `TypedDict` where the `evidence` field
carries an `Annotated[list[Evidence], operator.add]` annotation — that single
line is what makes parallel results combine instead of clobber each other.

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
7. **Supervisor** → **Judge**, which issues the final structured `Verdict`.
8. The CLI pretty-prints the verdict, the evidence summary, the credibility
   scores, and the citations.

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | **LangGraph** | `StateGraph`, `Send`, conditional edges, reducers |
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

No key is needed for Wikipedia or the local RAG model.

---

## Usage

```bash
# Verify any claim
python run.py "The Great Wall of China is visible from space."

# Run a built-in sample claim (1–6)
python run.py --example 2
python run.py --example 5

# List the sample claims
python run.py --list

# Watch each agent step stream by
python run.py --verbose "5G networks spread COVID-19."
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
├── config.py       # env vars, model names, runtime knobs
├── schemas.py      # Pydantic models: Evidence, ResearchPlan, Verdict, ...
├── tools.py        # web search (Tavily→DDG), Wikipedia, RAG retriever
├── agents.py       # the 5 agents + supervisor router
└── graph.py        # VerifactState, build_graph() — the orchestration
run.py              # CLI entrypoint
examples.py         # 6 sample claims for demos
tests/
└── test_smoke.py   # offline tests (no API key needed)
requirements.txt
.env.example
README.md
```

---

## Design decisions & trade-offs

- **Why a supervisor instead of a fixed pipeline?** A pipeline can't adapt: if
  the first research round returns almost nothing, a pipeline would still march
  straight to a confident-looking verdict. The supervisor's "go research more"
  branch is the difference between a robust system and a theatrical one.
- **Why two analysts instead of one?** Content and credibility are orthogonal.
  Ten blogs all parroting the same rumour look like "strong agreement" to a
  single analyst; splitting credibility into its own agent keeps that trap
  visible in the final score.
- **Why parallel research?** Latency and breadth. Sub-questions are independent,
  so running them sequentially would just waste time, and the cheap 8B model is
  fast enough to run several concurrently under Groq.
- **Why RAG over the evidence?** Researchers return noisy, sometimes huge blobs.
  Embedding them and retrieving only the chunks relevant to the *original* claim
  keeps the analyst focused and the prompt small — a small, deliberate use of
  the Week 2 RAG material inside an agentic system.
- **Why structured outputs everywhere?** Free-text handoffs between agents are
  fragile (one agent paraphrases a field name and the next one can't find it).
  Pydantic schemas make the contract explicit and the pipeline debuggable.
- **Why graceful degradation?** A grader or contest judge should be able to run
  this with just a Groq key. If Tavily isn't set, we use DuckDuckGo; if the
  embedding model can't download, we use raw snippets. Nothing should hard-fail
  on a missing optional dependency.

---

## Limitations & future work

- **Search quality is the ceiling.** With DuckDuckGo alone, rate-limiting and
  thin results are common; adding a Tavily key materially improves verdicts.
- **No claim/source image understanding** — text only.
- **Single language** (English) — the prompts and sources are English-centric.
- **No memory across runs.** A natural next step is a checkpointer
  (`langgraph-checkpoint-sqlite`) so Verifact can remember prior verdicts and
  avoid re-researching identical sub-questions.
- **Possible additions:** a "freshness" agent that biases toward recent sources
  for news claims; a hierarchical supervisor for multi-claim batches; exposing
  the graph over a FastAPI endpoint or a Streamlit UI.

---

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
- [LangChain — structured outputs](https://python.langchain.com/docs/how_to/structured_output/)
- [ChatGroq integration](https://python.langchain.com/docs/integrations/chat/groq/)
- [GroqCloud supported models](https://console.groq.com/docs/models)

---

## License

MIT — see the repo. Built as a capstone for the Agentic AI, Learners' Space 2026 bootcamp.
