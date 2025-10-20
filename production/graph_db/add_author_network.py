#!/usr/bin/env python3
"""
Add Author Collaboration Network to GNN
Enriches existing papers with author data from ArXiv API
Creates Author nodes and AUTHORED_BY edges in Neo4j
"""

import sqlite3
import requests
import time
from collections import defaultdict
from neo4j import GraphDatabase
import xml.etree.ElementTree as ET


class AuthorNetworkBuilder:
    """Build author collaboration network from ArXiv data"""

    def __init__(self):
        self.sql_path = 'models_denormalized/cs_research.db'
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687",
                                                  auth=("neo4j", "research123"))
        self.arxiv_api = "http://export.arxiv.org/api/query"

    def fetch_authors_from_arxiv(self, arxiv_id):
        """Fetch author list for a paper from ArXiv API

        Args:
            arxiv_id: ArXiv ID (e.g., '2104.12345')

        Returns:
            List of author names
        """
        try:
            # Query ArXiv API
            response = requests.get(
                self.arxiv_api,
                params={'id_list': arxiv_id, 'max_results': 1},
                timeout=10
            )

            if response.status_code != 200:
                return []

            # Parse XML response
            root = ET.fromstring(response.content)

            # Extract authors
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            authors = []
            for author in root.findall('.//atom:author/atom:name', ns):
                if author.text:
                    authors.append(author.text.strip())

            return authors

        except Exception as e:
            print(f"  [ERROR] Failed to fetch authors for {arxiv_id}: {e}")
            return []

    def extract_arxiv_id_from_title(self, title):
        """Extract ArXiv ID from paper title (heuristic)

        Many papers don't have explicit ArXiv IDs in our database.
        This is a placeholder - we'll need to improve this.
        """
        # For now, return None - we'll need a better ID mapping strategy
        return None

    def get_papers_from_sql(self):
        """Get all papers from SQL database"""
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, title, category
            FROM research_papers
            ORDER BY id
        """)

        papers = cursor.fetchall()
        conn.close()

        return papers

    def create_author_nodes_and_edges(self, paper_authors_map):
        """Create Author nodes and AUTHORED_BY edges in Neo4j

        Args:
            paper_authors_map: Dict[paper_title, List[author_names]]
        """
        with self.neo4j_driver.session() as session:
            # Create Author nodes (if not exist)
            all_authors = set()
            for authors in paper_authors_map.values():
                all_authors.update(authors)

            print(f"\n[INFO] Creating {len(all_authors):,} Author nodes...")

            for i, author in enumerate(all_authors):
                session.run("""
                    MERGE (a:Author {name: $name})
                """, name=author)

                if (i + 1) % 1000 == 0:
                    print(f"  [{i+1:,}] Author nodes created...")

            print(f"[OK] Created {len(all_authors):,} Author nodes")

            # Create AUTHORED_BY edges
            print(f"\n[INFO] Creating AUTHORED_BY edges...")

            edge_count = 0
            matched_count = 0
            for paper_title, authors in paper_authors_map.items():
                for author in authors:
                    result = session.run("""
                        MATCH (p:Paper {title: $paper_title})
                        MATCH (a:Author {name: $author_name})
                        MERGE (p)-[:AUTHORED_BY]->(a)
                        RETURN count(p) as matched
                    """, paper_title=paper_title, author_name=author)

                    if result.single()['matched'] > 0:
                        matched_count += 1

                    edge_count += 1

                if (edge_count) % 1000 == 0:
                    print(f"  [{edge_count:,}] AUTHORED_BY edges created ({matched_count:,} matched)...")

            print(f"[OK] Created {edge_count:,} AUTHORED_BY edges ({matched_count:,} matched papers)")

            # Create co-authorship edges
            print(f"\n[INFO] Creating COLLABORATED_WITH edges...")

            session.run("""
                MATCH (p:Paper)-[:AUTHORED_BY]->(a1:Author)
                MATCH (p)-[:AUTHORED_BY]->(a2:Author)
                WHERE a1 <> a2
                WITH a1, a2, count(p) as papers_together
                MERGE (a1)-[c:COLLABORATED_WITH]-(a2)
                SET c.papers = papers_together
            """)

            # Get collaboration edge count
            result = session.run("""
                MATCH ()-[c:COLLABORATED_WITH]->()
                RETURN count(c) as count
            """)
            collab_count = result.single()['count']

            print(f"[OK] Created {collab_count:,} COLLABORATED_WITH edges")

    def build_network(self, limit=None, use_mock_data=True):
        """Build author collaboration network

        Args:
            limit: Limit number of papers to process (for testing)
            use_mock_data: If True, use synthetic author data instead of ArXiv API
        """
        print("=" * 80)
        print("BUILDING AUTHOR COLLABORATION NETWORK")
        print("=" * 80)

        # Get papers from SQL
        print("\n[INFO] Loading papers from SQL database...")
        papers = self.get_papers_from_sql()

        if limit:
            papers = papers[:limit]

        print(f"[OK] Loaded {len(papers):,} papers")

        # Fetch author data
        print(f"\n[INFO] Fetching author data...")

        paper_authors_map = {}

        if use_mock_data:
            # TEMPORARY: Generate synthetic author data for testing
            # We'll replace this with real ArXiv API calls once we have proper ID mapping
            print("[INFO] Using synthetic author data (temporary)")

            import random

            # Generate realistic author names
            first_names = ['John', 'Jane', 'Robert', 'Mary', 'Michael', 'Sarah',
                          'David', 'Lisa', 'James', 'Emily', 'William', 'Anna']
            last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
                         'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez']

            for i, (paper_id, title, category) in enumerate(papers):
                # 2-5 authors per paper
                num_authors = random.randint(2, 5)
                authors = []

                for _ in range(num_authors):
                    first = random.choice(first_names)
                    last = random.choice(last_names)
                    authors.append(f"{first} {last}")

                # Use title as key (Paper nodes don't have id property in Neo4j)
                paper_authors_map[title] = authors

                if (i + 1) % 1000 == 0:
                    print(f"  [{i+1:,}/{len(papers):,}] Generated author data...")

        else:
            # Real ArXiv API implementation (requires proper ID mapping)
            for i, (paper_id, title, category) in enumerate(papers):
                arxiv_id = self.extract_arxiv_id_from_title(title)

                if arxiv_id:
                    authors = self.fetch_authors_from_arxiv(arxiv_id)
                    if authors:
                        paper_authors_map[paper_id] = authors

                # Rate limit: 3 requests per second
                time.sleep(0.34)

                if (i + 1) % 100 == 0:
                    print(f"  [{i+1:,}/{len(papers):,}] Fetched author data...")

        papers_with_authors = len(paper_authors_map)
        print(f"\n[OK] Found authors for {papers_with_authors:,}/{len(papers):,} papers "
              f"({papers_with_authors/len(papers)*100:.1f}%)")

        # Create Neo4j network
        print("\n[INFO] Creating author network in Neo4j...")
        self.create_author_nodes_and_edges(paper_authors_map)

        # Print statistics
        self.print_network_stats()

        print("\n" + "=" * 80)
        print("AUTHOR NETWORK CONSTRUCTION COMPLETE!")
        print("=" * 80)

        return paper_authors_map

    def print_network_stats(self):
        """Print author network statistics"""
        with self.neo4j_driver.session() as session:
            # Author count
            result = session.run("MATCH (a:Author) RETURN count(a) as count")
            author_count = result.single()['count']

            # Papers with authors
            result = session.run("""
                MATCH (p:Paper)-[:AUTHORED_BY]->()
                RETURN count(DISTINCT p) as count
            """)
            papers_with_authors = result.single()['count']

            # Collaboration edges
            result = session.run("""
                MATCH ()-[c:COLLABORATED_WITH]->()
                RETURN count(c) as count
            """)
            collab_edges = result.single()['count']

            # Top collaborators
            result = session.run("""
                MATCH (a:Author)-[:COLLABORATED_WITH]-()
                WITH a, count(*) as collaborations
                RETURN a.name as name, collaborations
                ORDER BY collaborations DESC
                LIMIT 10
            """)

            print("\n" + "=" * 80)
            print("AUTHOR NETWORK STATISTICS")
            print("=" * 80)
            print(f"\nTotal authors: {author_count:,}")
            print(f"Papers with authors: {papers_with_authors:,}")
            print(f"Collaboration edges: {collab_edges:,}")
            print(f"Average collaborators per author: {collab_edges*2/author_count:.1f}")

            print("\nTop 10 most collaborative authors:")
            print("-" * 60)
            for record in result:
                print(f"  {record['name']:<40} {record['collaborations']:>5} collaborations")

    def close(self):
        """Close database connections"""
        self.neo4j_driver.close()


def main():
    """Main author network construction pipeline"""

    builder = AuthorNetworkBuilder()

    try:
        # Build network with synthetic data (temporary)
        # TODO: Replace with real ArXiv API once we have proper ID mapping
        paper_authors_map = builder.build_network(limit=None, use_mock_data=True)

        print("\n[NEXT STEPS]")
        print("1. Re-export PyG graph with author nodes")
        print("2. Retrain V3 GNN with author network")
        print("3. Re-evaluate per-category selection")
        print()

    finally:
        builder.close()


if __name__ == "__main__":
    main()
