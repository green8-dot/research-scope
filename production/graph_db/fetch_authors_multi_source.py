#!/usr/bin/env python3
"""
Multi-Source Author Fetcher for Research Papers
- ArXiv API for CS/Physics/Math papers (955 papers)
- Semantic Scholar API for multidisciplinary papers (30,173 papers)
"""

import requests
import time
import json
import re
import html
from pathlib import Path
from neo4j import GraphDatabase
import xml.etree.ElementTree as ET


class MultiSourceAuthorFetcher:
    """Fetch authors from ArXiv and Semantic Scholar APIs"""

    def __init__(self):
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687",
                                                  auth=("neo4j", "research123"))
        self.arxiv_api = "http://export.arxiv.org/api/query"
        self.semantic_scholar_api = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.cache_file = Path("production/graph_db/author_cache_multi_source.json")
        self.cache = self.load_cache()

    def load_cache(self):
        """Load cached author data"""
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}

    def save_cache(self):
        """Save author cache"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def clean_title(self, title):
        """Clean title for API search"""
        # Decode HTML entities
        title = html.unescape(title)
        # Remove HTML tags
        title = re.sub(r'<[^>]+>', '', title)
        # Remove LaTeX math
        title = re.sub(r'\$[^\$]+\$', '', title)
        title = re.sub(r'\$\$[^\$]+\$\$', '', title)
        # Remove LaTeX commands
        title = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', title)
        title = re.sub(r'\\[a-zA-Z]+', '', title)
        # Remove curly braces
        title = title.replace('{', '').replace('}', '')
        # Remove subscripts/superscripts
        title = re.sub(r'_\{[^}]*\}', '', title)
        title = re.sub(r'\^\{[^}]*\}', '', title)
        # Clean whitespace
        title = ' '.join(title.split())
        return title.strip()

    def fetch_from_arxiv(self, title):
        """Fetch authors from ArXiv API"""
        try:
            clean_title = self.clean_title(title)
            clean_title = clean_title.replace('"', '').replace("'", "").strip()

            if len(clean_title) < 10:
                return []

            response = requests.get(
                self.arxiv_api,
                params={'search_query': f'ti:"{clean_title}"', 'max_results': 1},
                timeout=15
            )

            if response.status_code != 200:
                return []

            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('.//atom:entry', ns)

            if entry is None:
                return []

            authors = []
            for author in entry.findall('.//atom:author/atom:name', ns):
                if author.text:
                    authors.append(author.text.strip())

            return authors

        except Exception as e:
            print(f"  [ERROR] ArXiv API failed for '{title[:50]}...': {e}")
            return []

    def fetch_from_semantic_scholar(self, title, max_retries=3):
        """Fetch authors from Semantic Scholar API with retry logic for rate limiting"""
        clean_title = self.clean_title(title)

        if len(clean_title) < 10:
            return []

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.semantic_scholar_api,
                    params={'query': clean_title, 'limit': 1, 'fields': 'authors,title'},
                    timeout=15
                )

                # Handle rate limiting with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                        time.sleep(wait_time)
                        continue
                    else:
                        return []  # Give up after retries

                if response.status_code != 200:
                    return []

                data = response.json()

                if not data.get('data'):
                    return []

                paper = data['data'][0]
                authors = []

                for author in paper.get('authors', []):
                    author_name = author.get('name')
                    if author_name:
                        authors.append(author_name)

                return authors

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return []

        return []

    def get_papers_from_neo4j(self):
        """Get all papers from Neo4j with ArXiv tag info"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (p:Paper)
                RETURN
                    p.title AS title,
                    'ArxivPaper' IN labels(p) AS is_arxiv
                ORDER BY p.title
            """)

            papers = [(record['title'], record['is_arxiv']) for record in result]
            return papers

    def fetch_all_authors(self, limit=None, batch_size=100):
        """Fetch authors for all papers using appropriate API"""
        print("=" * 80)
        print("MULTI-SOURCE AUTHOR FETCHING")
        print("=" * 80)

        papers = self.get_papers_from_neo4j()

        if limit:
            papers = papers[:limit]

        arxiv_papers = [p for p in papers if p[1]]
        non_arxiv_papers = [p for p in papers if not p[1]]

        print(f"\n[INFO] Total papers: {len(papers):,}")
        print(f"[INFO] ArXiv papers: {len(arxiv_papers):,}")
        print(f"[INFO] Non-ArXiv papers: {len(non_arxiv_papers):,}")
        print(f"[INFO] Cache contains {len(self.cache):,} entries\n")

        paper_authors_map = {}
        arxiv_found = 0
        arxiv_not_found = 0
        semantic_found = 0
        semantic_not_found = 0

        start_time = time.time()

        for i, (title, is_arxiv) in enumerate(papers):
            # Check cache
            if title in self.cache:
                authors = self.cache[title]
                if authors:
                    paper_authors_map[title] = authors
                    if is_arxiv:
                        arxiv_found += 1
                    else:
                        semantic_found += 1

                if (i + 1) % 500 == 0:
                    print(f"  [{i+1:,}/{len(papers):,}] Cached: {title[:60]}...")
                continue

            # Fetch from appropriate API
            if is_arxiv:
                authors = self.fetch_from_arxiv(title)
                if authors:
                    arxiv_found += 1
                    if (i + 1) % 10 == 0:
                        print(f"  [{i+1:,}] ArXiv found {len(authors)} authors: {title[:50]}...")
                else:
                    arxiv_not_found += 1
                time.sleep(0.34)  # ArXiv rate limit: 3 req/sec
            else:
                authors = self.fetch_from_semantic_scholar(title)
                if authors:
                    semantic_found += 1
                    if (i + 1) % 10 == 0:
                        print(f"  [{i+1:,}] SemanticScholar found {len(authors)} authors: {title[:50]}...")
                else:
                    semantic_not_found += 1
                time.sleep(0.1)  # Semantic Scholar: 10 req/sec (very conservative)

            # Store results
            self.cache[title] = authors
            if authors:
                paper_authors_map[title] = authors

            # Save cache periodically
            if (i + 1) % batch_size == 0:
                self.save_cache()
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = len(papers) - (i + 1)
                eta = remaining / rate if rate > 0 else 0

                print(f"\n[CHECKPOINT] Cache saved ({len(self.cache):,} entries)")
                print(f"  Progress: {(i+1)/len(papers)*100:.1f}%")
                print(f"  ArXiv: {arxiv_found:,} found, {arxiv_not_found:,} not found")
                print(f"  Semantic Scholar: {semantic_found:,} found, {semantic_not_found:,} not found")
                print(f"  Rate: {rate:.2f} papers/sec | ETA: {eta/60:.1f} min\n")

        # Final save
        self.save_cache()

        elapsed = time.time() - start_time
        total_found = arxiv_found + semantic_found
        total_not_found = arxiv_not_found + semantic_not_found

        print(f"\n[OK] Completed in {elapsed/60:.1f} minutes")
        print(f"\n=== RESULTS ===")
        print(f"ArXiv API:")
        print(f"  Found: {arxiv_found:,}/{arxiv_found+arxiv_not_found:,} ({arxiv_found/(arxiv_found+arxiv_not_found)*100:.1f}%)")
        print(f"Semantic Scholar API:")
        print(f"  Found: {semantic_found:,}/{semantic_found+semantic_not_found:,} ({semantic_found/(semantic_found+semantic_not_found)*100 if (semantic_found+semantic_not_found) > 0 else 0:.1f}%)")
        print(f"Total:")
        print(f"  Found: {total_found:,}/{len(papers):,} ({total_found/len(papers)*100:.1f}%)")

        return paper_authors_map

    def create_author_network(self, paper_authors_map):
        """Create Author nodes and collaboration edges in Neo4j"""
        print(f"\n[INFO] Clearing existing author data...")
        with self.neo4j_driver.session() as session:
            session.run("MATCH (a:Author) DETACH DELETE a")

        with self.neo4j_driver.session() as session:
            # Create Author nodes
            all_authors = set()
            for authors in paper_authors_map.values():
                all_authors.update(authors)

            print(f"\n[INFO] Creating {len(all_authors):,} Author nodes...")
            for i, author in enumerate(all_authors):
                session.run("MERGE (a:Author {name: $name})", name=author)
                if (i + 1) % 1000 == 0:
                    print(f"  [{i+1:,}] Author nodes created...")

            print(f"[OK] Created {len(all_authors):,} Author nodes")

            # Create AUTHORED_BY edges
            print(f"\n[INFO] Creating AUTHORED_BY edges...")
            edge_count = 0
            for paper_title, authors in paper_authors_map.items():
                for author in authors:
                    session.run("""
                        MATCH (p:Paper {title: $paper_title})
                        MATCH (a:Author {name: $author_name})
                        MERGE (p)-[:AUTHORED_BY]->(a)
                    """, paper_title=paper_title, author_name=author)
                    edge_count += 1

                if edge_count % 1000 == 0:
                    print(f"  [{edge_count:,}] AUTHORED_BY edges created...")

            print(f"[OK] Created {edge_count:,} AUTHORED_BY edges")

            # Create COLLABORATED_WITH edges
            print(f"\n[INFO] Creating COLLABORATED_WITH edges...")
            session.run("""
                MATCH (p:Paper)-[:AUTHORED_BY]->(a1:Author)
                MATCH (p)-[:AUTHORED_BY]->(a2:Author)
                WHERE a1 <> a2
                WITH a1, a2, count(p) as papers_together
                MERGE (a1)-[c:COLLABORATED_WITH]-(a2)
                SET c.papers = papers_together
            """)

            result = session.run("MATCH ()-[c:COLLABORATED_WITH]->() RETURN count(c) as count")
            collab_count = result.single()['count']
            print(f"[OK] Created {collab_count:,} COLLABORATED_WITH edges")

    def close(self):
        """Close connections"""
        self.neo4j_driver.close()


def main():
    """Main pipeline"""
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if limit:
        print(f"\n[TEST MODE] Processing {limit} papers only\n")

    fetcher = MultiSourceAuthorFetcher()

    try:
        # Fetch authors from multiple sources
        paper_authors_map = fetcher.fetch_all_authors(limit=limit)

        if not paper_authors_map:
            print("\n[ERROR] No authors found!")
            return

        # Create Neo4j author network
        print("\n[INFO] Creating author network in Neo4j...")
        fetcher.create_author_network(paper_authors_map)

        print("\n" + "=" * 80)
        print("MULTI-SOURCE AUTHOR NETWORK COMPLETE!")
        print("=" * 80)

    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
