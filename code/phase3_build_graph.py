"""
Digital Detective — Phase 3: Neo4j Knowledge Graph Builder
===========================================================
Loads all extracted triples into Neo4j as a knowledge graph.
Each entity becomes a node, each relationship becomes an edge.

Setup:
    1. Install Neo4j Desktop: https://neo4j.com/download
    2. Create a local DBMS, set a password, click Start
    3. pip install neo4j
    4. Update NEO4J_PASSWORD below
    5. python phase3_build_graph.py

What gets built:
    - Nodes  : ThreatActor, Malware, Tool, CVE, IPAddress,
               Domain, Industry, Country, TTP, Campaign, etc.
    - Edges  : USES, TARGETS, EXPLOITS, ATTRIBUTED_TO, etc.
    - Indexes: fast lookup by name and type
"""

import json
from pathlib import Path
from neo4j import GraphDatabase
from project_config import get_env

# ── CONFIG ────────────────────────────────────────────────────────────────────
NEO4J_URI = get_env("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = get_env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_env("NEO4J_PASSWORD", "")

TRIPLES_FILE   = Path("digital_detective_data/all_triples.json")

# ── Driver ────────────────────────────────────────────────────────────────────

def get_driver():
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. Create .env from .env.example and set it."
        )
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("  Connected to Neo4j successfully")
        return driver
    except Exception as e:
        print(f"\n  ERROR: Cannot connect to Neo4j — {e}")
        print("  Make sure Neo4j Desktop is running and the password is correct.\n")
        raise

# ── Schema setup ──────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "ThreatActor", "Malware", "Tool", "CVE", "IPAddress",
    "Domain", "Industry", "Country", "TTP", "Campaign",
    "Organization", "FileHash", "Vulnerability"
]

def create_indexes(driver):
    """Create uniqueness constraints and indexes for fast lookup."""
    print("  Creating indexes …")
    with driver.session() as session:
        for etype in ENTITY_TYPES:
            try:
                session.run(
                    f"CREATE CONSTRAINT {etype.lower()}_name IF NOT EXISTS "
                    f"FOR (n:{etype}) REQUIRE n.name IS UNIQUE"
                )
            except Exception:
                pass  # constraint may already exist
    print("  Indexes ready")


def clear_graph(driver):
    """Wipe all nodes and edges (fresh start)."""
    print("  Clearing existing graph …")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  Graph cleared")

# ── Node + edge creation ──────────────────────────────────────────────────────

MERGE_NODE_QUERY = """
MERGE (n:{label} {{name: $name}})
ON CREATE SET n.name = $name, n.type = $type, n.created = timestamp()
ON MATCH  SET n.type = $type
RETURN n
"""

MERGE_EDGE_QUERY = """
MATCH (a:{slabel} {{name: $sname}})
MATCH (b:{olabel} {{name: $oname}})
MERGE (a)-[r:{predicate}]->(b)
ON CREATE SET r.confidence = $confidence,
              r.source_file = $source_file,
              r.source_page = $source_page,
              r.created = timestamp()
ON MATCH  SET r.confidence = CASE
                WHEN r.confidence < $confidence THEN $confidence
                ELSE r.confidence END
"""

def load_triples(driver, triples: list[dict]) -> dict:
    """Load all triples into Neo4j in batches."""
    stats = {"nodes_created": 0, "edges_created": 0, "errors": 0}

    print(f"  Loading {len(triples)} triples into Neo4j …")

    # Batch in groups of 100 for speed
    batch_size = 100
    batches = [triples[i:i+batch_size] for i in range(0, len(triples), batch_size)]

    from tqdm import tqdm
    for batch in tqdm(batches, desc="  Importing", ncols=70, unit="batch"):
        with driver.session() as session:
            for t in batch:
                try:
                    # Create subject node
                    session.run(
                        MERGE_NODE_QUERY.format(label=t["subject_type"]),
                        name=t["subject"],
                        type=t["subject_type"]
                    )
                    # Create object node
                    session.run(
                        MERGE_NODE_QUERY.format(label=t["object_type"]),
                        name=t["object"],
                        type=t["object_type"]
                    )
                    # Create relationship
                    session.run(
                        MERGE_EDGE_QUERY.format(
                            slabel=t["subject_type"],
                            olabel=t["object_type"],
                            predicate=t["predicate"]
                        ),
                        sname=t["subject"],
                        oname=t["object"],
                        confidence=t.get("confidence", 0.8),
                        source_file=t.get("source_file", ""),
                        source_page=t.get("source_page", 0)
                    )
                    stats["edges_created"] += 1
                except Exception as e:
                    stats["errors"] += 1

    return stats


# ── Graph statistics ──────────────────────────────────────────────────────────

def get_graph_stats(driver) -> dict:
    """Query the graph for summary statistics."""
    stats = {}
    with driver.session() as session:

        # Total nodes
        r = session.run("MATCH (n) RETURN count(n) AS c")
        stats["total_nodes"] = r.single()["c"]

        # Total edges
        r = session.run("MATCH ()-[r]->() RETURN count(r) AS c")
        stats["total_edges"] = r.single()["c"]

        # Nodes by type
        r = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS type, count(n) AS count
            ORDER BY count DESC
        """)
        stats["nodes_by_type"] = {row["type"]: row["count"] for row in r}

        # Top 10 most connected threat actors
        r = session.run("""
            MATCH (n:ThreatActor)-[r]->()
            RETURN n.name AS actor, count(r) AS connections
            ORDER BY connections DESC LIMIT 10
        """)
        stats["top_threat_actors"] = [
            {"name": row["actor"], "connections": row["connections"]} for row in r
        ]

        # Top 10 most targeted industries/countries
        r = session.run("""
            MATCH ()-[:TARGETS]->(t)
            RETURN t.name AS target, t.type AS type, count(*) AS times
            ORDER BY times DESC LIMIT 10
        """)
        stats["top_targets"] = [
            {"name": row["target"], "type": row["type"], "times": row["times"]} for row in r
        ]

        # Top malware families
        r = session.run("""
            MATCH (m:Malware)<-[r]-()
            RETURN m.name AS malware, count(r) AS mentions
            ORDER BY mentions DESC LIMIT 10
        """)
        stats["top_malware"] = [
            {"name": row["malware"], "mentions": row["mentions"]} for row in r
        ]

    return stats


def print_stats(stats: dict):
    print(f"\n  {'='*55}")
    print(f"  KNOWLEDGE GRAPH BUILT SUCCESSFULLY")
    print(f"  {'='*55}")
    print(f"  Total nodes  : {stats['total_nodes']}")
    print(f"  Total edges  : {stats['total_edges']}")

    print(f"\n  Nodes by type:")
    for ntype, count in stats["nodes_by_type"].items():
        bar = "█" * min(count // 5, 30)
        print(f"    {ntype:<15} {count:>4}  {bar}")

    print(f"\n  Top threat actors (by connections):")
    for a in stats["top_threat_actors"]:
        print(f"    {a['connections']:>3} edges — {a['name']}")

    print(f"\n  Top malware families:")
    for m in stats["top_malware"]:
        print(f"    {m['mentions']:>3} mentions — {m['name']}")

    print(f"\n  Most targeted:")
    for t in stats["top_targets"]:
        print(f"    {t['times']:>3}x  [{t['type']}]  {t['name']}")

    print(f"\n  Open Neo4j Browser: http://localhost:7474")
    print(f"  Try this query to explore:")
    print(f"    MATCH (n:ThreatActor)-[r]->(m) RETURN n,r,m LIMIT 50")
    print(f"\n  Next: run phase4_gnn_train.py")
    print(f"  {'='*55}\n")


# ── Sample Cypher queries ─────────────────────────────────────────────────────

def run_sample_queries(driver):
    """Run and print a few interesting queries as a demo."""
    print(f"\n  Sample intelligence queries:")
    print(f"  {'─'*45}")

    queries = [
        (
            "What malware does APT1 use?",
            "MATCH (a:ThreatActor)-[:USES]->(m:Malware) "
            "WHERE a.name CONTAINS 'APT1' RETURN a.name, m.name LIMIT 10"
        ),
        (
            "Which threat actors target Energy sector?",
            "MATCH (a:ThreatActor)-[:TARGETS]->(i:Industry) "
            "WHERE i.name CONTAINS 'Energy' OR i.name CONTAINS 'energy' "
            "RETURN a.name, i.name LIMIT 10"
        ),
        (
            "What does Stuxnet exploit?",
            "MATCH (m:Malware)-[:EXPLOITS]->(c) "
            "WHERE m.name CONTAINS 'Stuxnet' RETURN m.name, c.name LIMIT 10"
        ),
        (
            "Top 5 most mentioned countries?",
            "MATCH ()-[]->(c:Country) RETURN c.name, count(*) AS n "
            "ORDER BY n DESC LIMIT 5"
        ),
    ]

    with driver.session() as session:
        for question, cypher in queries:
            print(f"\n  Q: {question}")
            try:
                result = session.run(cypher)
                rows = result.data()
                if rows:
                    for row in rows[:5]:
                        vals = list(row.values())
                        print(f"     → {' | '.join(str(v) for v in vals)}")
                else:
                    print("     → No results")
            except Exception as e:
                print(f"     → Error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not TRIPLES_FILE.exists():
        print(f"\n  ERROR: {TRIPLES_FILE} not found.")
        print("  Run phase2_extract_triples.py first.\n")
        return

    triples = json.loads(TRIPLES_FILE.read_text(encoding="utf-8"))
    print(f"\n  Phase 3 — Neo4j Knowledge Graph")
    print(f"  {'─'*40}")
    print(f"  Triples to load : {len(triples)}")
    print(f"  Neo4j URI       : {NEO4J_URI}")
    print()

    driver = get_driver()

    clear_graph(driver)
    create_indexes(driver)
    load_stats = load_triples(driver, triples)

    print(f"\n  Import done — {load_stats['edges_created']} edges created, "
          f"{load_stats['errors']} errors")

    stats = get_graph_stats(driver)
    print_stats(stats)
    run_sample_queries(driver)

    driver.close()


if __name__ == "__main__":
    main()