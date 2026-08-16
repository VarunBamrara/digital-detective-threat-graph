"""
Digital Detective — Phase 4 (Improved): GNN Link Prediction
============================================================
Improvements over v1:
  1. Cleans noisy entity names before training
  2. Richer node features: type + degree + neighbor-type distribution (29-dim)
  3. Deeper 3-layer GraphSAGE
  4. 200 epochs + cosine LR scheduler + best-checkpoint saving
  5. Smarter missing-link candidates (actor→malware, actor→target only)

Usage:
    python phase4_gnn_train.py
"""

import json
import random
import numpy as np
from pathlib import Path
from neo4j import GraphDatabase
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from project_config import get_env

# ── CONFIG ────────────────────────────────────────────────────────────────────
NEO4J_URI = get_env("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = get_env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_env("NEO4J_PASSWORD", "")

GNN_DIR        = Path("digital_detective_data/gnn")
EPOCHS         = 200
HIDDEN_DIM     = 128
LEARNING_RATE  = 0.005
TEST_RATIO     = 0.15
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Noise filter ──────────────────────────────────────────────────────────────
NOISE_NAMES = {
    "apt", "threatactor", "threat actor", "attackers", "attacker",
    "unknown", "malware", "threat sources", "threat group", "actors",
    "actor", "group", "organization", "the attackers", "adversary",
    "adversaries", "they", "it", "victim", "victims", "target", "targets",
    "n/a", "none", "various", "multiple", "unidentified", "unidentified hackers",
    "10 different industries", "contractors"
}

def is_noise(name: str) -> bool:
    if not name or len(name.strip()) < 3:
        return True
    return name.strip().lower() in NOISE_NAMES

# ── Export from Neo4j ─────────────────────────────────────────────────────────

def export_from_neo4j():
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. Create .env from .env.example and set it."
        )
    print("  Connecting to Neo4j …")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    nodes, edges = {}, []
    with driver.session() as session:
        result = session.run(
            "MATCH (n) RETURN elementId(n) AS id, labels(n)[0] AS type, n.name AS name"
        )
        for row in result:
            nodes[row["id"]] = {"name": row["name"] or "", "type": row["type"] or "Unknown"}
        result = session.run(
            "MATCH (a)-[r]->(b) RETURN elementId(a) AS src, elementId(b) AS dst, type(r) AS rel"
        )
        for row in result:
            edges.append((row["src"], row["dst"], row["rel"]))
    driver.close()

    clean_nodes    = {k: v for k, v in nodes.items() if not is_noise(v["name"])}
    clean_node_ids = set(clean_nodes.keys())
    clean_edges    = [(s, d, r) for s, d, r in edges
                      if s in clean_node_ids and d in clean_node_ids]

    print(f"  Raw       : {len(nodes)} nodes, {len(edges)} edges")
    print(f"  Cleaned   : {len(clean_nodes)} nodes, {len(clean_edges)} edges")
    print(f"  Removed   : {len(nodes)-len(clean_nodes)} noisy nodes")
    return clean_nodes, clean_edges

# ── Build PyG graph ───────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "ThreatActor", "Malware", "Tool", "CVE", "IPAddress",
    "Domain", "Industry", "Country", "TTP", "Campaign",
    "Organization", "FileHash", "Vulnerability", "Unknown"
]
TYPE_TO_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}

def build_pyg_graph(nodes, edges):
    neo4j_ids    = sorted(nodes.keys())
    neo4j_to_pyg = {nid: i for i, nid in enumerate(neo4j_ids)}
    pyg_to_meta  = {i: nodes[nid] for nid, i in neo4j_to_pyg.items()}
    n = len(neo4j_ids)

    src_list, dst_list = [], []
    for src_neo, dst_neo, _ in edges:
        if src_neo in neo4j_to_pyg and dst_neo in neo4j_to_pyg:
            src_list.append(neo4j_to_pyg[src_neo])
            dst_list.append(neo4j_to_pyg[dst_neo])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Feature 1: one-hot entity type (14-dim)
    type_feat = torch.zeros(n, len(ENTITY_TYPES))
    for pid, meta in pyg_to_meta.items():
        tidx = TYPE_TO_IDX.get(meta["type"], TYPE_TO_IDX["Unknown"])
        type_feat[pid, tidx] = 1.0

    # Feature 2: normalized degree (1-dim)
    degree = torch.zeros(n, 1)
    for s, d in zip(src_list, dst_list):
        degree[s] += 1; degree[d] += 1
    degree = degree / (degree.max() + 1e-8)

    # Feature 3: neighbor type distribution (14-dim)
    nbr_type = torch.zeros(n, len(ENTITY_TYPES))
    for s, d in zip(src_list, dst_list):
        dtidx = TYPE_TO_IDX.get(pyg_to_meta[d]["type"], TYPE_TO_IDX["Unknown"])
        nbr_type[s, dtidx] += 1
    nbr_type = nbr_type / (nbr_type.sum(dim=1, keepdim=True).clamp(min=1))

    x    = torch.cat([type_feat, degree, nbr_type], dim=1)   # 29-dim
    data = Data(x=x, edge_index=edge_index, num_nodes=n)
    print(f"  PyG graph : {data.num_nodes} nodes | {data.num_edges} edges | {data.num_node_features} features")
    return data, pyg_to_meta

# ── Train/test split ──────────────────────────────────────────────────────────

def split(data, test_ratio=0.15):
    ne   = data.edge_index.shape[1]
    perm = torch.randperm(ne)
    ts   = int(ne * test_ratio)

    train_edge = data.edge_index[:, perm[ts:]]
    test_edge  = data.edge_index[:, perm[:ts]]

    existing = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    n = data.num_nodes
    ns, nd = [], []
    while len(ns) < ts * 2:
        s, d = random.randint(0, n-1), random.randint(0, n-1)
        if s != d and (s, d) not in existing:
            ns.append(s); nd.append(d)
    neg_edge = torch.tensor([ns[:ts], nd[:ts]], dtype=torch.long)
    print(f"  Split     : {train_edge.shape[1]} train | {test_edge.shape[1]} test+ | {neg_edge.shape[1]} test-")
    return train_edge, test_edge, neg_edge

# ── Model ─────────────────────────────────────────────────────────────────────

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, h):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h)
        self.conv2 = SAGEConv(h, h)
        self.conv3 = SAGEConv(h, h // 2)
        self.pred  = torch.nn.Sequential(
            torch.nn.Linear(h, h // 2), torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(h // 2, 1)
        )

    def encode(self, x, ei):
        x = self.conv1(x, ei).relu()
        x = F.dropout(x, 0.2, self.training)
        x = self.conv2(x, ei).relu()
        x = F.dropout(x, 0.2, self.training)
        x = self.conv3(x, ei)
        return x

    def decode(self, z, ei):
        return self.pred(torch.cat([z[ei[0]], z[ei[1]]], dim=-1)).squeeze()

    def forward(self, x, ei, pred_ei):
        return self.decode(self.encode(x, ei), pred_ei)

# ── Training ──────────────────────────────────────────────────────────────────

def train(data, train_edge, test_edge, neg_edge):
    model     = GraphSAGE(data.num_node_features, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    existing  = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    n         = data.num_nodes
    history   = []
    best_auc  = 0.0

    print(f"\n  Training GraphSAGE — {EPOCHS} epochs, hidden={HIDDEN_DIM} …")

    for epoch in tqdm(range(1, EPOCHS + 1), desc="  Training", ncols=72):
        model.train(); optimizer.zero_grad()
        pos = model(data.x, train_edge, train_edge)

        needed = train_edge.shape[1]
        ns2, nd2 = [], []
        while len(ns2) < needed:
            s, d = random.randint(0, n-1), random.randint(0, n-1)
            if s != d and (s, d) not in existing:
                ns2.append(s); nd2.append(d)
        neg_tr = torch.tensor([ns2[:needed], nd2[:needed]], dtype=torch.long)
        neg    = model(data.x, train_edge, neg_tr)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]),
            torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        )
        loss.backward(); optimizer.step(); scheduler.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                ps  = torch.sigmoid(model(data.x, train_edge, test_edge)).numpy()
                ns_ = torch.sigmoid(model(data.x, train_edge, neg_edge)).numpy()
            mn  = min(len(ps), len(ns_))
            auc = roc_auc_score(
                np.concatenate([np.ones(mn), np.zeros(mn)]),
                np.concatenate([ps[:mn], ns_[:mn]])
            )
            history.append({"epoch": epoch, "loss": round(loss.item(), 4), "auc": round(auc, 4)})
            tqdm.write(f"  Epoch {epoch:>3} | Loss {loss.item():.4f} | AUC {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), GNN_DIR / "model_best.pt")

    model.load_state_dict(torch.load(GNN_DIR / "model_best.pt"))
    print(f"\n  Best AUC during training: {best_auc:.4f}")
    return model, history

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, data, train_edge, test_edge, neg_edge):
    model.eval()
    with torch.no_grad():
        ps  = torch.sigmoid(model(data.x, train_edge, test_edge)).numpy()
        ns_ = torch.sigmoid(model(data.x, train_edge, neg_edge)).numpy()
    mn  = min(len(ps), len(ns_))
    yt  = np.concatenate([np.ones(mn), np.zeros(mn)])
    ys  = np.concatenate([ps[:mn], ns_[:mn]])
    auc = roc_auc_score(yt, ys)
    ap  = average_precision_score(yt, ys)
    grade = "EXCELLENT ✓" if auc >= 0.80 else "GOOD ✓" if auc >= 0.70 else "ACCEPTABLE"
    print(f"\n  {'='*45}")
    print(f"  EVALUATION")
    print(f"  AUC-ROC          : {auc:.4f}  (target >= 0.70)")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  Grade            : {grade}")
    return {"auc_roc": round(auc, 4), "average_precision": round(ap, 4)}

# ── Predict missing links ─────────────────────────────────────────────────────

def predict_links(model, data, meta, top_k=20):
    existing = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    actors   = [i for i, m in meta.items() if m["type"] == "ThreatActor" and not is_noise(m["name"])]
    malware  = [i for i, m in meta.items() if m["type"] in ("Malware", "Tool", "CVE") and not is_noise(m["name"])]
    targets  = [i for i, m in meta.items() if m["type"] in ("Industry", "Country", "Organization") and not is_noise(m["name"])]

    cs, cd = [], []
    for src in actors:
        for dst in random.sample(malware, min(60, len(malware))):
            if (src, dst) not in existing: cs.append(src); cd.append(dst)
        for dst in random.sample(targets, min(60, len(targets))):
            if (src, dst) not in existing: cs.append(src); cd.append(dst)

    if not cs:
        return []

    cand = torch.tensor([cs, cd], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        scores = torch.sigmoid(model.decode(z, cand)).numpy()

    ranked = sorted(zip(cs, cd, scores.tolist()), key=lambda x: x[2], reverse=True)[:top_k]
    preds  = []
    print(f"\n  Top {top_k} predicted missing links:")
    print(f"  {'─'*60}")
    for src, dst, score in ranked:
        sm, dm = meta.get(src, {}), meta.get(dst, {})
        p = {"subject": sm.get("name","?"), "subject_type": sm.get("type","?"),
             "object":  dm.get("name","?"), "object_type":  dm.get("type","?"),
             "score": round(score, 4)}
        preds.append(p)
        print(f"  {score:.3f}  [{p['subject_type']:11}] {p['subject'][:22]:<22} → [{p['object_type']:8}] {p['object'][:25]}")
    return preds

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    GNN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Phase 4 — GNN Link Prediction (Improved)")
    print(f"  {'─'*42}")

    nodes, edges           = export_from_neo4j()
    data, meta             = build_pyg_graph(nodes, edges)
    torch.save(data, GNN_DIR / "graph_data.pt")
    (GNN_DIR / "node_mapping.json").write_text(
        json.dumps({str(k): v for k, v in meta.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    train_edge, test_edge, neg_edge = split(data, TEST_RATIO)
    model, history                  = train(data, train_edge, test_edge, neg_edge)
    torch.save(model.state_dict(), GNN_DIR / "model.pt")

    metrics = evaluate(model, data, train_edge, test_edge, neg_edge)
    preds   = predict_links(model, data, meta, top_k=20)

    results = {
        "graph_stats":     {"nodes": data.num_nodes, "edges": data.num_edges, "features": data.num_node_features},
        "training":        {"epochs": EPOCHS, "hidden_dim": HIDDEN_DIM, "history": history},
        "evaluation":      metrics,
        "top_predictions": preds
    }
    (GNN_DIR / "gnn_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (GNN_DIR / "predictions.json").write_text(
        json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  {'='*45}")
    print(f"  PHASE 4 COMPLETE")
    print(f"  AUC-ROC  : {metrics['auc_roc']}  {'✓' if metrics['auc_roc'] >= 0.70 else '~'}")
    print(f"  Avg Prec : {metrics['average_precision']}")
    print(f"  Next     : python phase5_dashboard.py")
    print(f"  {'='*45}\n")

if __name__ == "__main__":
    main()