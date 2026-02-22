#!/usr/bin/env python3
"""Build a comprehensive knowledge graph from multiple open-source datasets.

Sources:
1. Wikidata simplified dump (via SPARQL endpoint) — real-world entities
2. Hardcoded ConceptNet-style commonsense triples
3. Our curated AI/ML domain knowledge

This creates a rich, real-world knowledge graph without depending on any
external API being available at build time.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
OUTPUT_PATH = GRAPH_DIR / "knowledge_graph.json"

# ═══════════════════════════════════════════════════════════
# 1. WIKIDATA SPARQL — Real-world entities & relationships
# ═══════════════════════════════════════════════════════════

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

WIKIDATA_QUERIES = {
    "programming_languages": """
        SELECT ?langLabel ?designerLabel ?paradigmLabel WHERE {
          ?lang wdt:P31 wd:Q9143.
          OPTIONAL { ?lang wdt:P178 ?designer. }
          OPTIONAL { ?lang wdt:P3966 ?paradigm. }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 40
    """,
    "ai_models": """
        SELECT ?modelLabel ?developerLabel WHERE {
          ?model wdt:P31/wdt:P279* wd:Q107733840.
          OPTIONAL { ?model wdt:P178 ?developer. }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 30
    """,
    "tech_companies": """
        SELECT ?companyLabel ?founderLabel ?industryLabel WHERE {
          VALUES ?company {
            wd:Q95 wd:Q380 wd:Q312 wd:Q36159 wd:Q19523 wd:Q21043440
            wd:Q4692 wd:Q468449 wd:Q193326 wd:Q58555648
          }
          OPTIONAL { ?company wdt:P112 ?founder. }
          OPTIONAL { ?company wdt:P452 ?industry. }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
    """,
}


def query_wikidata(sparql: str) -> list[dict[str, Any]]:
    """Execute a SPARQL query against Wikidata."""
    params = urllib.parse.urlencode({"query": sparql, "format": "json"})
    url = f"{WIKIDATA_ENDPOINT}?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DSPyClassificationStudio/2.0 (educational project)",
            "Accept": "application/sparql-results+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result: list[dict[str, Any]] = data.get("results", {}).get("bindings", [])
        return result
    except Exception as exc:
        print(f"  WARN: Wikidata query failed: {exc}")
        return []


def fetch_wikidata_triples() -> tuple[list[dict], list[dict]]:
    """Fetch real-world triples from Wikidata."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_nodes: set[str] = set()

    def add_node(name: str, ntype: str):
        key = f"{ntype}:{name.lower()}"
        if key not in seen_nodes and name and not name.startswith("Q"):
            seen_nodes.add(key)
            nodes.append({"name": name, "type": ntype})

    def add_edge(src_name: str, src_type: str, tgt_name: str, tgt_type: str,
                 relation: str, context: str = ""):
        if src_name and tgt_name and not src_name.startswith("Q") and not tgt_name.startswith("Q"):
            edges.append({
                "source": {"name": src_name, "type": src_type},
                "target": {"name": tgt_name, "type": tgt_type},
                "relation": relation,
                "weight": 0.85,
                "context": context or f"Wikidata: {src_name} {relation} {tgt_name}",
            })

    # Programming languages
    print("  Fetching programming languages from Wikidata...")
    for row in query_wikidata(WIKIDATA_QUERIES["programming_languages"]):
        lang = row.get("langLabel", {}).get("value", "")
        designer = row.get("designerLabel", {}).get("value", "")
        paradigm = row.get("paradigmLabel", {}).get("value", "")
        if lang:
            add_node(lang, "LANGUAGE")
            if designer:
                add_node(designer, "PERSON")
                add_edge(lang, "LANGUAGE", designer, "PERSON", "designed_by",
                         f"Wikidata: {lang} designed by {designer}")
            if paradigm:
                add_node(paradigm, "CONCEPT")
                add_edge(lang, "LANGUAGE", paradigm, "CONCEPT", "supports_paradigm",
                         f"Wikidata: {lang} supports {paradigm}")

    # AI Models
    print("  Fetching AI models from Wikidata...")
    for row in query_wikidata(WIKIDATA_QUERIES["ai_models"]):
        model = row.get("modelLabel", {}).get("value", "")
        developer = row.get("developerLabel", {}).get("value", "")
        if model:
            add_node(model, "MODEL")
            if developer:
                add_node(developer, "ORG")
                add_edge(model, "MODEL", developer, "ORG", "developed_by",
                         f"Wikidata: {model} developed by {developer}")

    # Tech companies
    print("  Fetching tech companies from Wikidata...")
    for row in query_wikidata(WIKIDATA_QUERIES["tech_companies"]):
        company = row.get("companyLabel", {}).get("value", "")
        founder = row.get("founderLabel", {}).get("value", "")
        industry = row.get("industryLabel", {}).get("value", "")
        if company:
            add_node(company, "ORG")
            if founder:
                add_node(founder, "PERSON")
                add_edge(company, "ORG", founder, "PERSON", "founded_by",
                         f"Wikidata: {company} founded by {founder}")
            if industry:
                add_node(industry, "FIELD")
                add_edge(company, "ORG", industry, "FIELD", "operates_in",
                         f"Wikidata: {company} operates in {industry}")

    return nodes, edges


# ═══════════════════════════════════════════════════════════
# 2. COMMONSENSE TRIPLES — ConceptNet-style knowledge
# ═══════════════════════════════════════════════════════════

COMMONSENSE_TRIPLES = [
    # is_a hierarchy
    ("machine learning", "CONCEPT", "artificial intelligence", "FIELD", "is_a", 0.95),
    ("deep learning", "CONCEPT", "machine learning", "FIELD", "is_a", 0.95),
    ("neural network", "CONCEPT", "deep learning", "FIELD", "used_in", 0.9),
    ("classification", "TASK", "machine learning", "FIELD", "is_a", 0.9),
    ("clustering", "TASK", "machine learning", "FIELD", "is_a", 0.85),
    ("regression", "TASK", "machine learning", "FIELD", "is_a", 0.85),
    ("natural language processing", "FIELD", "artificial intelligence", "FIELD", "subfield_of", 0.95),
    ("computer vision", "FIELD", "artificial intelligence", "FIELD", "subfield_of", 0.95),
    ("robotics", "FIELD", "artificial intelligence", "FIELD", "subfield_of", 0.85),
    ("speech recognition", "TASK", "natural language processing", "FIELD", "belongs_to", 0.9),
    ("machine translation", "TASK", "natural language processing", "FIELD", "belongs_to", 0.9),
    ("text summarization", "TASK", "natural language processing", "FIELD", "belongs_to", 0.9),
    ("image recognition", "TASK", "computer vision", "FIELD", "belongs_to", 0.9),
    ("object detection", "TASK", "computer vision", "FIELD", "belongs_to", 0.9),

    # used_for
    ("algorithm", "CONCEPT", "problem solving", "CONCEPT", "used_for", 0.9),
    ("database", "TECHNOLOGY", "data storage", "CONCEPT", "used_for", 0.95),
    ("API", "CONCEPT", "software integration", "CONCEPT", "used_for", 0.85),
    ("encryption", "CONCEPT", "cybersecurity", "FIELD", "used_for", 0.9),
    ("compiler", "CONCEPT", "programming", "CONCEPT", "used_in", 0.9),
    ("version control", "CONCEPT", "software development", "CONCEPT", "used_in", 0.9),
    ("Git", "TECHNOLOGY", "version control", "CONCEPT", "is_a", 0.95),
    ("Docker", "TECHNOLOGY", "containerisation", "CONCEPT", "used_for", 0.95),
    ("Kubernetes", "TECHNOLOGY", "container orchestration", "CONCEPT", "used_for", 0.95),

    # has_property
    ("Python", "LANGUAGE", "dynamically typed", "CONCEPT", "has_property", 0.9),
    ("Python", "LANGUAGE", "interpreted", "CONCEPT", "has_property", 0.9),
    ("Rust", "LANGUAGE", "memory safe", "CONCEPT", "has_property", 0.95),
    ("Rust", "LANGUAGE", "systems programming", "CONCEPT", "used_for", 0.9),
    ("JavaScript", "LANGUAGE", "web development", "CONCEPT", "used_for", 0.95),
    ("SQL", "LANGUAGE", "database querying", "CONCEPT", "used_for", 0.95),

    # related_to
    ("training data", "CONCEPT", "machine learning", "FIELD", "required_by", 0.9),
    ("overfitting", "CONCEPT", "machine learning", "FIELD", "problem_in", 0.85),
    ("bias", "CONCEPT", "artificial intelligence", "FIELD", "problem_in", 0.85),
    ("GPU", "TECHNOLOGY", "deep learning", "FIELD", "accelerates", 0.95),
    ("tensor", "CONCEPT", "deep learning", "FIELD", "data_structure_of", 0.85),
    ("gradient", "CONCEPT", "neural network", "CONCEPT", "used_in", 0.9),
    ("loss function", "CONCEPT", "neural network", "CONCEPT", "component_of", 0.9),
    ("activation function", "CONCEPT", "neural network", "CONCEPT", "component_of", 0.9),
    ("epoch", "CONCEPT", "training", "CONCEPT", "measured_by", 0.8),
    ("batch size", "CONCEPT", "training", "CONCEPT", "parameter_of", 0.8),
    ("learning rate", "CONCEPT", "training", "CONCEPT", "parameter_of", 0.85),
    ("hyperparameter", "CONCEPT", "machine learning", "FIELD", "concept_in", 0.85),

    # Science/math foundations
    ("linear algebra", "FIELD", "machine learning", "FIELD", "foundation_of", 0.9),
    ("calculus", "FIELD", "machine learning", "FIELD", "foundation_of", 0.85),
    ("probability", "FIELD", "machine learning", "FIELD", "foundation_of", 0.9),
    ("statistics", "FIELD", "data science", "FIELD", "foundation_of", 0.9),
    ("information theory", "FIELD", "machine learning", "FIELD", "foundation_of", 0.8),
    ("optimisation", "CONCEPT", "machine learning", "FIELD", "used_in", 0.9),
    ("Bayes theorem", "CONCEPT", "probability", "FIELD", "part_of", 0.9),

    # Software engineering
    ("unit test", "CONCEPT", "software quality", "CONCEPT", "ensures", 0.85),
    ("CI/CD", "CONCEPT", "software deployment", "CONCEPT", "automates", 0.9),
    ("microservice", "CONCEPT", "software architecture", "CONCEPT", "pattern_of", 0.85),
    ("REST API", "CONCEPT", "web development", "CONCEPT", "pattern_of", 0.9),
    ("cloud computing", "FIELD", "software deployment", "CONCEPT", "enables", 0.9),
    ("serverless", "CONCEPT", "cloud computing", "FIELD", "paradigm_of", 0.8),
    ("open source", "CONCEPT", "software development", "CONCEPT", "methodology_of", 0.85),

    # Data concepts
    ("data pipeline", "CONCEPT", "data engineering", "FIELD", "tool_in", 0.9),
    ("ETL", "CONCEPT", "data pipeline", "CONCEPT", "is_a", 0.9),
    ("data lake", "CONCEPT", "data storage", "CONCEPT", "is_a", 0.85),
    ("data warehouse", "CONCEPT", "data storage", "CONCEPT", "is_a", 0.85),
    ("feature engineering", "CONCEPT", "machine learning", "FIELD", "technique_in", 0.9),
    ("data augmentation", "CONCEPT", "machine learning", "FIELD", "technique_in", 0.85),
    ("cross-validation", "CONCEPT", "machine learning", "FIELD", "technique_in", 0.9),
]


def build_commonsense_triples() -> tuple[list[dict], list[dict]]:
    """Build nodes and edges from our commonsense triples."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()

    for src, src_type, tgt, tgt_type, relation, weight in COMMONSENSE_TRIPLES:
        for name, ntype in [(src, src_type), (tgt, tgt_type)]:
            key = f"{ntype}:{name.lower()}"
            if key not in seen:
                seen.add(key)
                nodes.append({"name": name, "type": ntype})

        edges.append({
            "source": {"name": src, "type": src_type},
            "target": {"name": tgt, "type": tgt_type},
            "relation": relation,
            "weight": weight,
            "context": f"Commonsense: {src} {relation} {tgt}",
        })

    return nodes, edges


# ═══════════════════════════════════════════════════════════
# 3. MERGE
# ═══════════════════════════════════════════════════════════

def merge_graphs(
    existing_path: Path,
    *graphs: tuple[list[dict], list[dict]],
) -> dict:
    """Merge multiple (nodes, edges) pairs with existing graph."""
    # Load existing
    existing: dict[str, Any] = {"nodes": [], "edges": []}
    if existing_path.exists():
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    node_keys: set[str] = set()
    edge_keys: set[str] = set()
    all_nodes: list[dict] = []
    all_edges: list[dict] = []

    # Process existing
    for n in existing.get("nodes", []):
        key = f"{n.get('type', 'CONCEPT').upper()}:{n['name'].lower()}"
        if key not in node_keys:
            node_keys.add(key)
            all_nodes.append(n)

    for e in existing.get("edges", []):
        src = e.get("source", {})
        tgt = e.get("target", {})
        src_name = src.get("name", "") if isinstance(src, dict) else str(src)
        tgt_name = tgt.get("name", "") if isinstance(tgt, dict) else str(tgt)
        rel = e.get("relation", "")
        key = f"{src_name.lower()}|{rel.lower()}|{tgt_name.lower()}"
        if key not in edge_keys:
            edge_keys.add(key)
            all_edges.append(e)

    # Merge new graphs
    for nodes, edges in graphs:
        for n in nodes:
            key = f"{n.get('type', 'CONCEPT').upper()}:{n['name'].lower()}"
            if key not in node_keys:
                node_keys.add(key)
                all_nodes.append(n)

        added = 0
        for e in edges:
            src = e.get("source", {})
            tgt = e.get("target", {})
            src_name = src.get("name", "") if isinstance(src, dict) else str(src)
            tgt_name = tgt.get("name", "") if isinstance(tgt, dict) else str(tgt)
            rel = e.get("relation", "")
            key = f"{src_name.lower()}|{rel.lower()}|{tgt_name.lower()}"
            if key not in edge_keys:
                edge_keys.add(key)
                all_edges.append(e)
                added += 1

        print(f"  Added {added} new edges from source")

    return {
        "nodes": all_nodes,
        "edges": all_edges,
        "node_count": len(all_nodes),
        "edge_count": len(all_edges),
    }


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building knowledge graph from open-source data")
    print("=" * 60)

    # 1. Wikidata
    print("\n[1/2] Wikidata real-world entities...")
    wikidata_graph = fetch_wikidata_triples()
    print(f"  Got {len(wikidata_graph[0])} nodes, {len(wikidata_graph[1])} edges")

    # 2. Commonsense
    print("\n[2/2] Commonsense knowledge triples...")
    commonsense_graph = build_commonsense_triples()
    print(f"  Got {len(commonsense_graph[0])} nodes, {len(commonsense_graph[1])} edges")

    # Merge all
    print("\nMerging with existing curated graph...")
    merged = merge_graphs(OUTPUT_PATH, wikidata_graph, commonsense_graph)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nFinal graph: {merged['node_count']} nodes, {merged['edge_count']} edges")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


