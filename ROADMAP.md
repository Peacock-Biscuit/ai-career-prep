# Roadmap
## DS → MLE Transition Study Plan
### Focus: LLM-Augmented Search + Causal Inference
**Background:** Data Analyst / Scientist  
**Timeline:** 6 months  
**Target roles:** Search Relevance MLE · Experimentation / Causal ML Engineer · Ranking & Personalization MLE

---

## Why This Combination

| Pillar | What you bring (DS) | What to build (MLE) |
|---|---|---|
| LLM + Search | Familiarity with NLP basics | Bi-encoders, RAG, query rewriting at scale |
| Causal Inference | A/B testing, DiD, PSM | IPS, CLTR, OPE, Double ML in production |
| Engineering | Notebooks, pandas, SQL | Production Python, APIs, Docker, serving |

---

## Skill Gap Overview

### Already strong — leverage these
- Python & pandas
- SQL & data wrangling
- ML fundamentals (regression, trees, evaluation)
- A/B experimentation design
- Statistical thinking & data storytelling

### Gaps to close
- Production software engineering (OOP, testing, CI/CD)
- ML system design (retrieval + ranking architecture)
- Model serving & APIs (FastAPI, Docker, latency budgets)
- Two-tower models, dense retrieval, FAISS
- Inverse propensity scoring, CLTR, offline policy evaluation
- Distributed training, LLM inference optimization

---

## 6-Month Roadmap

### Phase 1 — Engineer First (Months 1–2)
**Goal:** Close the software engineering gap. This is the #1 thing separating DS from MLE.

#### What to learn
- Production Python: classes, type hints, `pytest`, `mypy`
- OOP design patterns used in ML systems (factory, strategy)
- Git workflows, code review, CI/CD basics
- Data structures & algorithms (LeetCode medium difficulty)
- REST APIs with FastAPI — wrap a model in a real endpoint
- Docker: containerize your first ML project

#### What to do
- [ ] Solve 3 LeetCode mediums per week (arrays, hashmaps, graphs)
- [ ] Refactor a past DS notebook into a proper Python package with tests
- [ ] Deploy a simple sklearn model as a FastAPI service in Docker
- [ ] Read: *Designing ML Systems* ch 1–4 (Chip Huyen)

#### Resources
- [FastAPI docs](https://fastapi.tiangolo.com/)
- [Designing ML Systems — Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- LeetCode: focus on `array`, `hashmap`, `sliding window`, `BFS/DFS` patterns

---

### Phase 2 — Search Systems (Months 2–3)
**Goal:** Build deep expertise in modern search architecture.

#### What to learn
- BM25 internals: TF-IDF, IDF smoothing, k1/b parameter tuning
- Bi-encoder retrieval: training with in-batch negatives and hard negatives
- Cross-encoder re-ranking: when to use, latency tradeoffs
- FAISS index types: Flat, IVF-PQ, HNSW — recall vs. speed
- Hybrid search: combining sparse (BM25) + dense scores via RRF
- NDCG, MRR, MAP — and why they're biased by click data

#### What to do
- [ ] Build a hybrid BM25 + bi-encoder search engine on a public corpus
- [ ] Fine-tune a `sentence-transformer` on MS MARCO
- [ ] Read the DPR paper (Karpukhin et al., Facebook AI, 2020)
- [ ] Evaluate your model on the BEIR benchmark across 5+ datasets

#### Key papers
- **DPR** — Karpukhin et al. (2020): foundational bi-encoder retrieval
- **SPLADE-v3** — Piwowarski et al. (2024): sparse neural retrieval

---

### Phase 3 — Causal ML Depth (Months 3–4)
**Goal:** Leverage DS advantage. Deepen causal inference into the ML engineering context.

#### What to learn
- Position bias model: examination hypothesis, propensity estimation
- Inverse propensity scoring (IPS) for unbiased learning-to-rank
- Counterfactual learning to rank (CLTR) — Joachims framework
- Offline policy evaluation (OPE): IPS, DM, and doubly robust estimators
- Double ML (DML) — Chernozhukov et al., nuisance models with cross-fitting
- CUPED for A/B test variance reduction in search experiments

#### What to do
- [ ] Implement IPS-weighted LambdaMART on the MSLR-WEB10K dataset
- [ ] Compare biased vs. debiased model using simulated click data
- [ ] Read Joachims et al. (2017) position bias paper in depth
- [ ] Implement CUPED on a synthetic A/B test dataset

#### Key papers
- **Position-Bias Estimation with Randomization** — Joachims et al. (2018)
- **Unbiased LTR with Propensity-Aware Distillation** — Agarwal et al., Google (2020)
- **Doubly Robust Joint Learning for Recommendation** — Wang et al. (2019)

---

### Phase 4 — LLM Integration (Months 4–5)
**Goal:** Combine LLMs with your search and causal foundation into a unique, production-relevant skillset.

#### What to learn
- LLM query rewriting: prompt strategies, evaluation, caching
- RAG architecture: chunking, embedding, retrieval, generation pipeline
- Faithfulness evaluation: does the answer follow from retrieved docs?
- LLM-generated counterfactuals for debiased training data augmentation
- Double ML with LLM embeddings as nuisance estimators
- Calibration and hallucination detection in LLM-search pipelines

#### What to do
- [ ] Add LLM query rewriting to your search engine (project 2)
- [ ] Build a RAG pipeline with causal faithfulness evaluation (project 4)
- [ ] Implement Double ML on LaLonde dataset using LLM text features
- [ ] Write a technical blog post on causal debiasing of search

#### Key papers
- **GRIT** — Muennighoff et al. (2023): single LLM for generation + embedding
- **Causal Reasoning and LLMs** — Kıcıman et al., Microsoft (2023)
- **Double ML with Text Features** (2022): DML with NLP embeddings as nuisance models

---

### Phase 5 — Ship & Interview (Months 5–6)
**Goal:** Finalize portfolio, sharpen system design, and run a structured job search.

#### What to learn
- ML system design: 5-step framework (scope → data → model → serve → monitor)
- Behavioral stories in STAR format: projects, tradeoffs, failures, learnings
- LLM serving at scale: vLLM, KV cache, continuous batching

#### What to do
- [ ] Publish 2 projects on GitHub with clean READMEs and blog posts
- [ ] Do 5+ mock system design interviews (Pramp, Interviewing.io)
- [ ] Apply to 15 target companies — personalize for search/ranking roles
- [ ] Prepare a 90-second pitch: why this combination, why you, why now
- [ ] Reach out to search MLEs at target companies for informational chats

---

## Portfolio Projects

### Project 1 — Causal Debiased Search Ranker
**Type:** LLM + Causal  
**Dataset:** MSLR-WEB10K  
**Description:** Build a learning-to-rank model on click logs, apply IPS to correct position bias, and compare biased vs. debiased performance on held-out relevance judgments. Use an LLM to estimate propensity scores for unobserved positions.  
**Why it stands out:** Directly addresses the #1 real problem in production search. Most candidates can train a ranker; almost none can debias one.  
**Stack:** `LambdaMART` · `IPS/CLTR` · `PyTorch` · `FAISS` · `FastAPI` · `Docker`

---

### Project 2 — LLM-Powered Semantic Search with Causal Eval
**Type:** LLM  
**Dataset:** MS MARCO or custom corpus  
**Description:** Build a hybrid BM25 + dense retrieval engine. Use an LLM to rewrite queries and re-rank results. Evaluate using offline policy evaluation (OPE) on historical click data — not just naive NDCG.  
**Why it stands out:** Combining LLM retrieval with causally-correct offline eval signals applied scientist-level thinking.  
**Stack:** `sentence-transformers` · `BM25` · `OPE/IPS` · `OpenAI API` · `Elasticsearch`

---

### Project 3 — Treatment Effect Estimation with LLM Features
**Type:** LLM + Causal  
**Dataset:** LaLonde, ACIC benchmark, or synthetic  
**Description:** Apply Double ML where an LLM encodes text features (product descriptions, user reviews) as inputs to nuisance models. Estimate heterogeneous treatment effects (HTE) and visualize CATE across subgroups.  
**Why it stands out:** Directly bridges your DS causal background with LLMs — very few MLE candidates can demonstrate this.  
**Stack:** `EconML` · `DoWhy` · `Double ML` · `HuggingFace embeddings` · `SHAP`

---

### Project 4 — RAG System with Grounded Answer Attribution
**Type:** LLM  
**Description:** Build a RAG pipeline that retrieves, generates, and attributes each claim to a source passage. Add a causal faithfulness check: does removing a retrieved passage change the answer? Use this intervention as an eval metric.  
**Why it stands out:** Attribution + causal faithfulness eval is an open research problem. A publishable-quality idea.  
**Stack:** `LangChain` · `pgvector` · `causal faithfulness` · `FastAPI` · `Docker`

---

## Target Companies

| Company | Why relevant |
|---|---|
| Google (Search, YouTube) | Foundational search + LLM research, publishes on CLTR |
| Microsoft (Bing) | Heavy LLM-search integration post-GPT-4 |
| LinkedIn | Feed ranking + experimentation platform (publishes CUPED) |
| Airbnb | Search ranking with causal eval (published extensively) |
| Spotify | Personalization + search, strong causal inference culture |
| Amazon (Search, Ads) | Large-scale product search, IPS-based ranking |
| Perplexity / You.com / Exa | LLM-native search startups, move fast |
| Walmart Labs / eBay / Etsy | E-commerce search, less competitive than FAANG |

---

## Weekly Habit Checklist

- [ ] 3× LeetCode mediums (arrays, graphs, DP)
- [ ] 1 paper read and summarized (30 min)
- [ ] 2 hours on current portfolio project
- [ ] 1 system design practice session (alone or with a partner)
- [ ] 1 technical blog post or LinkedIn post per month

---

## Key Resources

### Books
- *Designing ML Systems* — Chip Huyen
- *Causal Inference: The Mixtape* — Scott Cunningham (free online)
- *The Effect* — Nick Huntington-Klein (free online)

### Courses
- Full Stack Deep Learning (FSDL) — free on YouTube
- Cassidy Williams' ML Engineering course
- fast.ai Practical Deep Learning

### Communities
- MLOps Community Slack
- Causal Inference reading groups (Twitter/X)
- Eugene Yan's blog (applied ML at Amazon)
- Hamel Husain's blog (LLM evaluation)
