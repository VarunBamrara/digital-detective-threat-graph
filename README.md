# The Digital Detective — Threat Intelligence Knowledge Graph

5-stage pipeline extracting structured entity-relation triples from APT threat
intelligence reports (8 reports, 314 pages, 1,464 chunks) via LLM extraction,
building a Neo4j knowledge graph, and training a GNN for link prediction.

- `code/phase1_extract_text.py` — PDF text extraction
- `code/phase2_extract_triples.py` — LLM-based triple extraction (Groq API)
- `code/phase3_build_graph.py` — Neo4j knowledge graph construction
- `code/phase4_gnn_train.py` — GNN training for link prediction
- `code/phase5_dashboard.py` — interactive dashboard
- `REPORT.pdf` — full project report

Requires a Groq API key and a local/cloud Neo4j instance — see `code/.env.example`.
No public live demo (requires private API keys and a graph database).
