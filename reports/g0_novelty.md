# G0 novelty record — EchoBase (EAB)

Rubric (plan §2.3): A1 rows re-segmented between versions; A2 LLM-derived values reused with a stated guarantee;
A3 column decomposability measured on real changes; A4 real workloads in ≥ 2 domains, all artifacts released.
Decision rule (fixed): a paper meeting A1 ∧ A2 → stop/re-scope (DEFT vision); meeting A1 or A2 alone → cite and
position precisely; meeting none → background.

## Pass 1 — 2026-10-28

Query strings used (web search, standard/extended):
1. "self-maintainable LLM-derived views decomposable semantic operators incremental maintenance without re-invoking LLM"
2. "incremental maintenance speech transcript diarization resegmentation derived database view LLM extraction 2026"
3. "incremental maintenance RAG index when document chunking changes re-chunking reuse embeddings LLM extractions"
4. "LLM-powered data processing pipeline incremental recomputation document updates DocETL LOTUS caching benchmark 2026"
5. earlier passes (2026-10-27): "incremental view maintenance LLM-extracted tables document updates semantic operators 2026";
   "LLM model version upgrade re-extraction cost materialized LLM outputs incremental recomputation data lake 2026 arXiv"

| Hit (opened) | A1 | A2 | A3 | A4 | Action |
|---|---|---|---|---|---|
| SemReuse, arXiv 2610.16035 (2026-10-12) — new predicates over fixed tuples, exact stratified audit | ✗ | ✓ | ✗ | ✗ | cite; reuse audit technique; position (PC chair co-author) |
| sIVM, Zenodo 21833641 (2026-08-07) — prompt edits, stable rows (cell computed "over its row") | ✗ | ✓ | ✗ | ✗ | cite; baseline "anchor-cert" models its stable-identity assumption |
| Cyclex, ICDE 2008 — IE recycling across text snapshots | partial | ✗ | ✗ | ✗ | cite as classic baseline |
| Enzyme, arXiv 2603.27775 — SQL IVM | ✗ | ✗ | ✗ | ✗ | background |
| Cost Scales with Change, arXiv 2608.16621 — synthetic SVD substrate | ✗ | ✗ | ✗ | ✗ | background |
| BLIP provenance, arXiv 2608.25210 — minimal input subset preserving an LLM answer | ✗ | ✗ | partial | ✗ | cite (related to sub-span sufficiency) |
| rag-go PR #72 / RAG reindexing guides — content-hash chunk reuse | partial | ✗ | ✗ | ✗ | motivates the hash baseline |
| DiarizationLM, CBAL (ACL SRW 2026) — LLM diarization correction | (produces re-segmentation) | ✗ | ✗ | ✗ | motivation: re-segmentation is routine |

**Outcome pass 1: no paper meets A1 ∧ A2. Continue.** Pass 2 scheduled 2026-11-07 with the same strings.
