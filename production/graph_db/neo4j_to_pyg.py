#!/usr/bin/env python3
"""
Export Neo4j graph to PyTorch Geometric format
Prepares data for GNN training with 31,128 papers + 8 categories
"""

import json
import torch
from pathlib import Path
from neo4j import GraphDatabase
from torch_geometric.data import Data
from collections import defaultdict


class Neo4jToPyGExporter:
    """Export Neo4j graph to PyTorch Geometric Data object"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="research123"):
        print(f"Connecting to Neo4j at {uri}...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("[OK] Connected to Neo4j\n")

    def close(self):
        self.driver.close()

    def export_graph_structure(self):
        """
        Export graph structure from Neo4j
        Returns: papers, categories, edges
        """
        print("=" * 80)
        print("EXPORTING GRAPH STRUCTURE FROM NEO4J")
        print("=" * 80)

        with self.driver.session() as session:
            # Step 1: Export all paper nodes
            print("\nStep 1: Exporting paper nodes...")
            result = session.run("""
                MATCH (p:Paper)
                RETURN id(p) AS neo4j_id, p.title AS title, p.abstract AS abstract
                ORDER BY id(p)
            """)

            papers = []
            neo4j_id_to_idx = {}  # Map Neo4j ID to PyG node index
            for idx, record in enumerate(result):
                papers.append({
                    'neo4j_id': record['neo4j_id'],
                    'pyg_idx': idx,
                    'title': record['title'],
                    'abstract': record['abstract']
                })
                neo4j_id_to_idx[record['neo4j_id']] = idx

            print(f"[OK] Exported {len(papers):,} paper nodes")

            # Step 2: Export all category nodes
            print("\nStep 2: Exporting category nodes...")
            result = session.run("""
                MATCH (c:Category)
                RETURN id(c) AS neo4j_id, c.name AS name
                ORDER BY c.name
            """)

            categories = []
            for idx, record in enumerate(result):
                category_idx = len(papers) + idx  # Categories come after papers
                categories.append({
                    'neo4j_id': record['neo4j_id'],
                    'pyg_idx': category_idx,
                    'name': record['name']
                })
                neo4j_id_to_idx[record['neo4j_id']] = category_idx

            print(f"[OK] Exported {len(categories)} category nodes")

            # Category name to index mapping (for labels)
            category_name_to_idx = {cat['name']: i for i, cat in enumerate(categories)}

            # Step 3: Export classification edges
            print("\nStep 3: Exporting CLASSIFIED_AS edges...")
            result = session.run("""
                MATCH (p:Paper)-[:CLASSIFIED_AS]->(c:Category)
                RETURN id(p) AS paper_id, id(c) AS category_id
            """)

            classification_edges = []
            for record in result:
                paper_idx = neo4j_id_to_idx[record['paper_id']]
                category_idx = neo4j_id_to_idx[record['category_id']]
                classification_edges.append((paper_idx, category_idx))

            print(f"[OK] Exported {len(classification_edges):,} classification edges")

            # Step 3.5: Export citation edges (CITES)
            print("\nStep 3.5: Exporting CITES edges...")
            result = session.run("""
                MATCH (p1:Paper)-[:CITES]->(p2:Paper)
                RETURN id(p1) AS citing_id, id(p2) AS cited_id
            """)

            citation_edges = []
            for record in result:
                citing_idx = neo4j_id_to_idx[record['citing_id']]
                cited_idx = neo4j_id_to_idx[record['cited_id']]
                citation_edges.append((citing_idx, cited_idx))

            print(f"[OK] Exported {len(citation_edges):,} citation edges")

            # Step 3.6: Export author-based paper-paper edges (CO_AUTHORED)
            print("\nStep 3.6: Exporting author collaboration edges...")
            result = session.run("""
                MATCH (p1:Paper)-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(p2:Paper)
                WHERE id(p1) < id(p2)
                RETURN id(p1) AS paper1_id, id(p2) AS paper2_id
            """)

            coauthor_edges = []
            for record in result:
                paper1_idx = neo4j_id_to_idx.get(record['paper1_id'])
                paper2_idx = neo4j_id_to_idx.get(record['paper2_id'])

                if paper1_idx is not None and paper2_idx is not None:
                    # Add bidirectional edges for co-authorship
                    coauthor_edges.append((paper1_idx, paper2_idx))
                    coauthor_edges.append((paper2_idx, paper1_idx))

            print(f"[OK] Exported {len(coauthor_edges):,} co-authorship edges")

            # Combine all edges for graph structure
            all_edges = classification_edges + citation_edges + coauthor_edges

            # Step 4: Create multi-label matrix for papers
            print("\nStep 4: Creating multi-label matrix...")
            paper_labels = torch.zeros(len(papers), len(categories), dtype=torch.float)

            for paper_idx, category_idx_global in classification_edges:
                # Convert global category index to category class index (0-7)
                category_class_idx = category_idx_global - len(papers)
                paper_labels[paper_idx, category_class_idx] = 1.0

            # Count multi-label papers
            multi_label_count = (paper_labels.sum(dim=1) > 1).sum().item()
            print(f"[OK] Multi-label matrix: {paper_labels.shape}")
            print(f"     Papers with multiple labels: {multi_label_count:,} ({multi_label_count/len(papers)*100:.1f}%)")

        print("\n" + "=" * 80)
        print("EXPORT SUMMARY")
        print("=" * 80)
        print(f"Paper nodes:          {len(papers):,}")
        print(f"Category nodes:       {len(categories)}")
        print(f"Total nodes:          {len(papers) + len(categories):,}")
        print(f"\nEdge Types:")
        print(f"  Classification edges:  {len(classification_edges):,}")
        print(f"  Citation edges:        {len(citation_edges):,}")
        print(f"  Co-authorship edges:   {len(coauthor_edges):,}")
        print(f"  Total edges:           {len(all_edges):,}")
        print(f"\nAverage categories/paper: {len(classification_edges)/len(papers):.2f}")
        print(f"\nCategory Distribution:")
        for cat in categories:
            cat_idx = cat['pyg_idx'] - len(papers)
            count = paper_labels[:, cat_idx].sum().int().item()
            print(f"  {cat['name']:10s}: {count:6,} papers")

        return {
            'papers': papers,
            'categories': categories,
            'edges': all_edges,
            'labels': paper_labels,
            'category_name_to_idx': category_name_to_idx
        }

    def create_pyg_data_object(self, graph_data, node_features=None):
        """
        Create PyTorch Geometric Data object from exported graph

        Args:
            graph_data: Dict from export_graph_structure()
            node_features: Optional pre-computed node features (papers + categories)
                          If None, placeholder features will be used

        Returns:
            torch_geometric.data.Data object
        """
        print("\n" + "=" * 80)
        print("CREATING PYTORCH GEOMETRIC DATA OBJECT")
        print("=" * 80)

        num_papers = len(graph_data['papers'])
        num_categories = len(graph_data['categories'])
        total_nodes = num_papers + num_categories

        # Create edge index [2, num_edges]
        edge_index = torch.tensor(graph_data['edges'], dtype=torch.long).t().contiguous()
        print(f"\n[OK] Edge index: {edge_index.shape}")

        # Create node features
        if node_features is None:
            print("[WARN] No node features provided, using placeholder features")
            # Placeholder: Will be replaced with SciBERT embeddings
            x = torch.randn(total_nodes, 768)  # 768-dim placeholder
        else:
            x = node_features
            print(f"[OK] Using provided node features: {x.shape}")

        # Labels (only for papers, not categories)
        y = graph_data['labels']  # [num_papers, num_categories]
        print(f"[OK] Labels: {y.shape}")

        # Create masks for train/val/test split (stratified by category)
        train_mask, val_mask, test_mask = self._create_train_val_test_masks(
            y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_papers=num_papers,
            num_categories=num_categories
        )

        print(f"\n[OK] PyG Data object created:")
        print(f"     Nodes: {data.num_nodes:,}")
        print(f"     Edges: {data.num_edges:,}")
        print(f"     Features: {data.x.shape}")
        print(f"     Labels: {data.y.shape}")
        print(f"     Train papers: {train_mask.sum():,}")
        print(f"     Val papers: {val_mask.sum():,}")
        print(f"     Test papers: {test_mask.sum():,}")

        return data

    def _create_train_val_test_masks(self, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Create stratified train/val/test masks for papers

        Ensures each split has similar category distribution
        """
        num_papers = labels.shape[0]

        # Initialize masks
        train_mask = torch.zeros(num_papers, dtype=torch.bool)
        val_mask = torch.zeros(num_papers, dtype=torch.bool)
        test_mask = torch.zeros(num_papers, dtype=torch.bool)

        # Random split (stratified by having at least one label)
        indices = torch.randperm(num_papers)

        train_end = int(num_papers * train_ratio)
        val_end = train_end + int(num_papers * val_ratio)

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True

        return train_mask, val_mask, test_mask

    def save_export(self, graph_data, output_dir):
        """Save exported graph data to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save graph structure as JSON
        print(f"\n[SAVE] Saving graph structure...")
        graph_metadata = {
            'num_papers': len(graph_data['papers']),
            'num_categories': len(graph_data['categories']),
            'num_edges': len(graph_data['edges']),
            'categories': [cat['name'] for cat in graph_data['categories']],
            'category_name_to_idx': graph_data['category_name_to_idx']
        }

        with open(output_dir / 'graph_metadata.json', 'w') as f:
            json.dump(graph_metadata, f, indent=2)

        # Save paper metadata (for reference)
        papers_metadata = [
            {'idx': p['pyg_idx'], 'title': p['title']}
            for p in graph_data['papers']
        ]
        with open(output_dir / 'papers_metadata.json', 'w') as f:
            json.dump(papers_metadata, f, indent=2)

        # Save labels
        torch.save(graph_data['labels'], output_dir / 'labels.pt')

        print(f"[OK] Saved to {output_dir}/")
        print(f"     - graph_metadata.json")
        print(f"     - papers_metadata.json")
        print(f"     - labels.pt")


def main():
    """Export Neo4j graph to PyG format"""
    print("=" * 80)
    print("NEO4J TO PYTORCH GEOMETRIC EXPORT")
    print("=" * 80)

    exporter = Neo4jToPyGExporter()

    try:
        # Export graph structure from Neo4j
        graph_data = exporter.export_graph_structure()

        # Create PyG Data object (without node features for now)
        # Node features will be generated separately with SciBERT
        pyg_data = exporter.create_pyg_data_object(graph_data)

        # Save export
        output_dir = Path("/media/d1337g/SystemBackup/framework_baseline/production/graph_db/cached_embeddings")
        exporter.save_export(graph_data, output_dir)

        # Save PyG Data object (placeholder features)
        torch.save(pyg_data, output_dir / 'graph_data_placeholder.pt')
        print(f"\n[OK] Saved PyG Data object: {output_dir}/graph_data_placeholder.pt")

        print("\n" + "=" * 80)
        print("EXPORT COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Generate SciBERT embeddings for papers and categories")
        print("  2. Replace placeholder features with SciBERT embeddings")
        print("  3. Train GraphSAGE model on final graph")

    finally:
        exporter.close()


if __name__ == "__main__":
    main()
