#!/usr/bin/env python3
"""Download a subset of ConceptNet (open-source commonsense KG) and merge with
our curated AI/ML knowledge graph.

ConceptNet is licensed under CC-BY-SA 4.0.
Source: https://conceptnet.io/
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
RAW_PATH = GRAPH_DIR / "conceptnet_raw.json"
MERGED_PATH = GRAPH_DIR / "knowledge_graph.json"

# Concepts to query — tech, AI/ML, science, and general knowledge
CONCEPTS = [
    "machine_learning", "artificial_intelligence", "neural_network",
    "deep_learning", "natural_language_processing", "computer_vision",
    "python", "programming", "algorithm", "data_science",
    "sentiment_analysis", "classification", "knowledge_graph", "transformer",
    "robot", "automation", "technology", "software", "database",
    "statistics", "mathematics", "science", "engineering",
    "internet", "cloud_computing", "cybersecurity",
    "language_model", "reinforcement_learning", "computer_science",
    "information_retrieval", "text_mining", "data_mining",
    "supervised_learning", "unsupervised_learning",
]

# Map ConceptNet relation labels to snake_case
RELATION_MAP = {
    "RelatedTo": "related_to",
    "IsA": "is_a",
    "PartOf": "part_of",
    "HasA": "has_a",
    "UsedFor": "used_for",
    "CapableOf": "capable_of",
    "AtLocation": "at_location",
    "Causes": "causes",
    "HasProperty": "has_property",
    "MadeOf": "made_of",
    "ReceivesAction": "receives_action",
    "CreatedBy": "created_by",
    "Synonym": "synonym",
    "Antonym": "antonym",
    "DistinctFrom": "distinct_from",
    "DerivedFrom": "derived_from",
    "SymbolOf": "symbol_of",
    "DefinedAs": "defined_as",
    "MannerOf": "manner_of",
    "LocatedNear": "located_near",
    "HasContext": "has_context",
    "SimilarTo": "similar_to",
    "EtymologicallyRelatedTo": "etymologically_related_to",
    "EtymologicallyDerivedFrom": "etymologically_derived_from",
    "CausesDesire": "causes_desire",
    "MotivatedByGoal": "motivated_by_goal",
    "ObstructedBy": "obstructed_by",
    "Desires": "desires",
    "HasSubevent": "has_subevent",
    "HasFirstSubevent": "has_first_subevent",
    "HasLastSubevent": "has_last_subevent",
    "HasPrerequisite": "has_prerequisite",
    "Entails": "entails",
    "FormOf": "form_of",
}


def fetch_concept(concept: str) -> list[dict[str, Any]]:
    """Fetch edges for a single concept from the ConceptNet API."""
    url = f"http://api.conceptnet.io/c/en/{concept}?limit=25"
    req = urllib.request.Request(
        url, headers={"User-Agent": "DSPyClassificationStudio/2.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result: list[dict[str, Any]] = data.get("edges", [])
        return result
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  WARN: skip {concept}: {exc}")
        return []


def parse_edges(raw_edges: list[dict]) -> tuple[set[str], list[dict]]:
    """Parse ConceptNet edges into our graph format."""
    nodes: set[str] = set()
    edges: list[dict] = []

    for edge in raw_edges:
        rel_label = edge.get("rel", {}).get("label", "")
        start = edge.get("start", {})
        end = edge.get("end", {})

        if start.get("language") != "en" or end.get("language") != "en":
            continue

        src = start.get("label", "").strip()
        tgt = end.get("label", "").strip()
        weight = edge.get("weight", 1.0)

        if not src or not tgt:
            continue
        if len(src) > 60 or len(tgt) > 60:
            continue
        # Skip very low-weight edges
        if weight < 1.0:
            continue

        relation = RELATION_MAP.get(rel_label, rel_label.lower())
        normalized_weight = round(min(weight, 10.0) / 10.0, 3)

        nodes.add(src)
        nodes.add(tgt)
        edges.append({
            "source": src,
            "target": tgt,
            "relation": relation,
            "weight": max(normalized_weight, 0.1),
            "context": f"ConceptNet: {src} {rel_label} {tgt}",
        })

    return nodes, edges


def download_conceptnet() -> dict:
    """Download and aggregate ConceptNet data."""
    all_nodes: set[str] = set()
    all_edges: list[dict] = []

    print(f"Downloading ConceptNet data for {len(CONCEPTS)} concepts...")
    for i, concept in enumerate(CONCEPTS):
        print(f"  [{i+1}/{len(CONCEPTS)}] {concept}...", end=" ", flush=True)
        raw = fetch_concept(concept)
        nodes, edges = parse_edges(raw)
        all_nodes.update(nodes)
        all_edges.extend(edges)
        print(f"{len(edges)} edges")
        time.sleep(0.3)  # Rate limiting

    # Deduplicate edges
    seen: set[str] = set()
    unique: list[dict] = []
    for e in all_edges:
        key = f"{e['source'].lower()}|{e['relation']}|{e['target'].lower()}"
        if key not in seen:
            seen.add(key)
            unique.append(e)

    print(f"\nTotal: {len(all_nodes)} nodes, {len(unique)} unique edges")
    return {
        "source": "ConceptNet 5.7 (CC-BY-SA 4.0)",
        "url": "https://conceptnet.io/",
        "nodes": sorted(list(all_nodes)),
        "edges": unique,
        "node_count": len(all_nodes),
        "edge_count": len(unique),
    }


def merge_with_existing(conceptnet_data: dict) -> dict:
    """Merge ConceptNet data with our curated AI/ML graph."""
    # Load existing curated graph
    existing_path = MERGED_PATH
    existing: dict[str, Any] = {"nodes": [], "edges": []}
    if existing_path.exists():
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # Build set of existing nodes
    existing_node_keys = set()
    for n in existing.get("nodes", []):
        existing_node_keys.add(f"{n.get('type', 'CONCEPT').upper()}:{n['name'].lower()}")

    # Add ConceptNet nodes (as CONCEPT type since ConceptNet doesn't have types)
    merged_nodes = list(existing.get("nodes", []))
    for node_name in conceptnet_data["nodes"]:
        key = f"CONCEPT:{node_name.lower()}"
        if key not in existing_node_keys:
            merged_nodes.append({"name": node_name, "type": "CONCEPT"})
            existing_node_keys.add(key)

    # Build set of existing edges
    existing_edge_keys = set()
    for e in existing.get("edges", []):
        src = e.get("source", {})
        tgt = e.get("target", {})
        src_name = src.get("name", "") if isinstance(src, dict) else str(src)
        tgt_name = tgt.get("name", "") if isinstance(tgt, dict) else str(tgt)
        rel = e.get("relation", "")
        existing_edge_keys.add(f"{src_name.lower()}|{rel.lower()}|{tgt_name.lower()}")

    # Add ConceptNet edges
    merged_edges = list(existing.get("edges", []))
    added = 0
    for e in conceptnet_data["edges"]:
        key = f"{e['source'].lower()}|{e['relation']}|{e['target'].lower()}"
        if key not in existing_edge_keys:
            merged_edges.append({
                "source": {"name": e["source"], "type": "CONCEPT"},
                "target": {"name": e["target"], "type": "CONCEPT"},
                "relation": e["relation"],
                "weight": e["weight"],
                "context": e["context"],
            })
            existing_edge_keys.add(key)
            added += 1

    print(f"Merged: added {added} ConceptNet edges to existing {len(existing.get('edges', []))} edges")

    return {
        "nodes": merged_nodes,
        "edges": merged_edges,
        "node_count": len(merged_nodes),
        "edge_count": len(merged_edges),
    }


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    conceptnet_data = download_conceptnet()

    # Save raw
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(conceptnet_data, f, indent=2, ensure_ascii=False)
    print(f"Saved raw ConceptNet data to {RAW_PATH}")

    # Merge with existing
    merged = merge_with_existing(conceptnet_data)

    # Save merged
    with open(MERGED_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Saved merged graph to {MERGED_PATH} ({merged['node_count']} nodes, {merged['edge_count']} edges)")


if __name__ == "__main__":
    main()


