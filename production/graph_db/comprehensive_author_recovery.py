#!/usr/bin/env python3
"""
Comprehensive Multi-Tier Author Recovery System
Target: 95%+ success rate through cascading fallback methods
"""

import requests
import time
import json
import re
from pathlib import Path
from typing import List, Optional


class ComprehensiveAuthorRecovery:
    """
    Multi-tier author recovery with intelligent fallbacks

    Tier 1: Free comprehensive APIs (OpenAlex, CrossRef, DBLP)
    Tier 2: Specialized APIs (PubMed, Europe PMC, IEEE)
    Tier 3: DOI resolution and metadata extraction
    Tier 4: Text parsing and pattern matching
    Tier 5: Manual lookup suggestions for critical failures
    """

    def __init__(self):
        self.cache_file = Path("production/graph_db/author_cache_multi_source.json")
        self.cache = self.load_cache()

        # API endpoints
        self.openalex_api = "https://api.openalex.org/works"
        self.crossref_api = "https://api.crossref.org/works"
        self.dblp_api = "https://dblp.org/search/publ/api"
        self.pubmed_api = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.europepmc_api = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

        # Success tracking by tier
        self.tier_stats = {
            'tier1_openalex': 0,
            'tier1_crossref': 0,
            'tier1_dblp': 0,
            'tier2_pubmed': 0,
            'tier2_europepmc': 0,
            'tier3_doi': 0,
            'tier4_text': 0,
            'final_failure': 0
        }

    def load_cache(self):
        """Load existing cache"""
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}

    def save_cache(self):
        """Save updated cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    # ==================== TIER 1: Free Comprehensive APIs ====================

    def tier1_openalex(self, title: str) -> Optional[List[str]]:
        """OpenAlex API - Very comprehensive, free"""
        try:
            response = requests.get(
                self.openalex_api,
                params={'filter': f'title.search:{title}', 'per-page': 1},
                headers={'User-Agent': 'mailto:research@example.com'},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    work = results[0]
                    authors = []
                    for authorship in work.get('authorships', []):
                        author = authorship.get('author', {})
                        author_name = author.get('display_name')
                        if author_name:
                            authors.append(author_name)

                    if authors:
                        self.tier_stats['tier1_openalex'] += 1
                        return authors

            time.sleep(0.2)  # 5 req/sec
        except Exception:
            pass
        return None

    def tier1_crossref(self, title: str) -> Optional[List[str]]:
        """CrossRef API - General academic papers, free"""
        try:
            response = requests.get(
                self.crossref_api,
                params={'query.title': title, 'rows': 1},
                headers={'User-Agent': 'mailto:research@example.com'},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                if items:
                    work = items[0]
                    authors = []
                    for author in work.get('author', []):
                        given = author.get('given', '')
                        family = author.get('family', '')
                        if given and family:
                            authors.append(f"{given} {family}")
                        elif family:
                            authors.append(family)

                    if authors:
                        self.tier_stats['tier1_crossref'] += 1
                        return authors

            time.sleep(0.1)  # 10 req/sec
        except Exception:
            pass
        return None

    def tier1_dblp(self, title: str) -> Optional[List[str]]:
        """DBLP API - Computer Science bibliography, excellent coverage"""
        try:
            response = requests.get(
                self.dblp_api,
                params={'q': title, 'format': 'json', 'h': 1},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                hits = data.get('result', {}).get('hits', {}).get('hit', [])
                if hits:
                    info = hits[0].get('info', {})
                    authors_data = info.get('authors', {}).get('author', [])

                    authors = []
                    if isinstance(authors_data, list):
                        for author in authors_data:
                            if isinstance(author, dict):
                                name = author.get('text', '')
                            else:
                                name = str(author)
                            if name:
                                authors.append(name)
                    elif isinstance(authors_data, dict):
                        name = authors_data.get('text', '')
                        if name:
                            authors.append(name)

                    if authors:
                        self.tier_stats['tier1_dblp'] += 1
                        return authors

            time.sleep(0.15)  # Conservative
        except Exception:
            pass
        return None

    # ==================== TIER 2: Specialized APIs ====================

    def tier2_pubmed(self, title: str) -> Optional[List[str]]:
        """PubMed API - Medical/biology papers"""
        try:
            # Search for paper
            search_response = requests.get(
                self.pubmed_api,
                params={
                    'db': 'pubmed',
                    'term': title,
                    'retmode': 'json',
                    'retmax': 1
                },
                timeout=15
            )

            if search_response.status_code == 200:
                search_data = search_response.json()
                id_list = search_data.get('esearchresult', {}).get('idlist', [])

                if id_list:
                    # Fetch details
                    pmid = id_list[0]
                    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    fetch_response = requests.get(
                        fetch_url,
                        params={'db': 'pubmed', 'id': pmid, 'retmode': 'json'},
                        timeout=15
                    )

                    if fetch_response.status_code == 200:
                        fetch_data = fetch_response.json()
                        result = fetch_data.get('result', {}).get(str(pmid), {})
                        authors_data = result.get('authors', [])

                        authors = []
                        for author in authors_data:
                            name = author.get('name', '')
                            if name:
                                authors.append(name)

                        if authors:
                            self.tier_stats['tier2_pubmed'] += 1
                            return authors

            time.sleep(0.35)  # PubMed rate limit: 3 req/sec
        except Exception:
            pass
        return None

    def tier2_europepmc(self, title: str) -> Optional[List[str]]:
        """Europe PMC API - European papers, biomedical focus"""
        try:
            response = requests.get(
                self.europepmc_api,
                params={'query': f'TITLE:"{title}"', 'format': 'json', 'pageSize': 1},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get('resultList', {}).get('result', [])

                if results:
                    paper = results[0]
                    author_string = paper.get('authorString', '')

                    if author_string:
                        # Parse author string (usually "Last1 First1, Last2 First2")
                        authors = [a.strip() for a in author_string.split(',')]
                        authors = [a for a in authors if a]

                        if authors:
                            self.tier_stats['tier2_europepmc'] += 1
                            return authors

            time.sleep(0.2)
        except Exception:
            pass
        return None

    # ==================== TIER 3: DOI Resolution ====================

    def tier3_doi_lookup(self, title: str) -> Optional[List[str]]:
        """
        Try to find DOI and resolve it
        Many papers have DOIs that can be resolved to full metadata
        """
        try:
            # Try CrossRef DOI search
            response = requests.get(
                self.crossref_api,
                params={'query.title': title, 'rows': 1},
                headers={'User-Agent': 'mailto:research@example.com'},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])

                if items:
                    doi = items[0].get('DOI')
                    if doi:
                        # Resolve DOI via doi.org
                        doi_response = requests.get(
                            f"https://doi.org/{doi}",
                            headers={'Accept': 'application/vnd.citationstyles.csl+json'},
                            timeout=15
                        )

                        if doi_response.status_code == 200:
                            metadata = doi_response.json()
                            authors_data = metadata.get('author', [])

                            authors = []
                            for author in authors_data:
                                given = author.get('given', '')
                                family = author.get('family', '')
                                if given and family:
                                    authors.append(f"{given} {family}")
                                elif family:
                                    authors.append(family)

                            if authors:
                                self.tier_stats['tier3_doi'] += 1
                                return authors

            time.sleep(0.15)
        except Exception:
            pass
        return None

    # ==================== TIER 4: Text Parsing ====================

    def tier4_text_extraction(self, title: str, abstract: str = "") -> Optional[List[str]]:
        """
        Extract author names from abstract or title patterns
        Some papers mention authors in acknowledgments or references

        Patterns like:
        - "by John Smith and Jane Doe"
        - "Authors: Smith J, Doe J"
        - Email patterns with names
        """
        try:
            # Pattern 1: "by [Name] and [Name]"
            pattern1 = r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+and\s+([A-Z][a-z]+\s+[A-Z][a-z]+))?'
            matches1 = re.findall(pattern1, abstract)

            # Pattern 2: "Authors?: [Name], [Name]"
            pattern2 = r'Authors?:\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:,\s+[A-Z][a-z]+\s+[A-Z][a-z]+)*)'
            matches2 = re.findall(pattern2, abstract)

            # Pattern 3: Email patterns (name@domain)
            pattern3 = r'([a-z]+\.[a-z]+)@'
            matches3 = re.findall(pattern3, abstract.lower())

            authors = []

            # Process pattern 1
            for match in matches1:
                for name in match:
                    if name:
                        authors.append(name)

            # Process pattern 2
            for match in matches2:
                names = match.split(',')
                authors.extend([n.strip() for n in names if n.strip()])

            # Process pattern 3 (convert email to name)
            for email_name in matches3:
                if '.' in email_name:
                    parts = email_name.split('.')
                    name = ' '.join([p.capitalize() for p in parts])
                    authors.append(name)

            if authors:
                # Deduplicate
                authors = list(set(authors))
                self.tier_stats['tier4_text'] += 1
                return authors
        except Exception:
            pass
        return None

    # ==================== Main Recovery Pipeline ====================

    def recover_authors_comprehensive(self, title: str, abstract: str = "") -> List[str]:
        """
        Execute full cascading fallback pipeline
        Returns authors from first successful tier
        """

        # TIER 1: Free comprehensive APIs (parallel concept - try all, take first success)
        methods_tier1 = [
            ('OpenAlex', self.tier1_openalex),
            ('CrossRef', self.tier1_crossref),
            ('DBLP', self.tier1_dblp)
        ]

        for name, method in methods_tier1:
            authors = method(title)
            if authors:
                return authors

        # TIER 2: Specialized APIs
        methods_tier2 = [
            ('PubMed', self.tier2_pubmed),
            ('EuropePMC', self.tier2_europepmc)
        ]

        for name, method in methods_tier2:
            authors = method(title)
            if authors:
                return authors

        # TIER 3: DOI resolution
        authors = self.tier3_doi_lookup(title)
        if authors:
            return authors

        # TIER 4: Text extraction (if abstract available)
        if abstract:
            authors = self.tier4_text_extraction(title, abstract)
            if authors:
                return authors

        # Final failure
        self.tier_stats['final_failure'] += 1
        return []

    def process_all_failures(self, batch_size=100):
        """Process all failed papers through comprehensive pipeline"""

        print("=" * 80)
        print("COMPREHENSIVE MULTI-TIER AUTHOR RECOVERY")
        print("=" * 80)

        # Get failed titles
        failed_titles = [title for title, authors in self.cache.items() if authors == []]

        print(f"\n[INFO] Target: {len(failed_titles):,} failed papers")
        print(f"[INFO] Goal: 95%+ recovery rate\n")

        print("[TIER 1] OpenAlex, CrossRef, DBLP")
        print("[TIER 2] PubMed, Europe PMC")
        print("[TIER 3] DOI resolution")
        print("[TIER 4] Text extraction\n")

        start_time = time.time()

        for i, title in enumerate(failed_titles):
            # Execute comprehensive recovery
            authors = self.recover_authors_comprehensive(title)

            # Update cache
            if authors:
                self.cache[title] = authors
                if (i + 1) % 10 == 0 or len(authors) > 0:
                    print(f"  [{i+1:,}/{len(failed_titles):,}] Found {len(authors)} authors: {title[:50]}...")
            else:
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1:,}/{len(failed_titles):,}] Still failed: {title[:50]}...")

            # Save periodically
            if (i + 1) % batch_size == 0:
                self.save_cache()
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = len(failed_titles) - (i + 1)
                eta = remaining / rate if rate > 0 else 0

                total_recovered = sum(v for k, v in self.tier_stats.items() if k != 'final_failure')

                print(f"\n[CHECKPOINT] {i+1:,}/{len(failed_titles):,} processed")
                print(f"  Tier 1 (OpenAlex):   {self.tier_stats['tier1_openalex']:,}")
                print(f"  Tier 1 (CrossRef):   {self.tier_stats['tier1_crossref']:,}")
                print(f"  Tier 1 (DBLP):       {self.tier_stats['tier1_dblp']:,}")
                print(f"  Tier 2 (PubMed):     {self.tier_stats['tier2_pubmed']:,}")
                print(f"  Tier 2 (EuropePMC):  {self.tier_stats['tier2_europepmc']:,}")
                print(f"  Tier 3 (DOI):        {self.tier_stats['tier3_doi']:,}")
                print(f"  Tier 4 (Text):       {self.tier_stats['tier4_text']:,}")
                print(f"  Total recovered:     {total_recovered:,} ({total_recovered/(i+1)*100:.1f}%)")
                print(f"  Still failing:       {self.tier_stats['final_failure']:,}")
                print(f"  Rate: {rate:.2f} papers/sec | ETA: {eta/60:.1f} min\n")

        # Final save
        self.save_cache()

        elapsed = time.time() - start_time
        total_recovered = sum(v for k, v in self.tier_stats.items() if k != 'final_failure')

        print(f"\n" + "=" * 80)
        print("COMPREHENSIVE RECOVERY COMPLETE")
        print("=" * 80)

        print(f"\n[RESULTS BY TIER]")
        print(f"  Tier 1 (OpenAlex):   {self.tier_stats['tier1_openalex']:,}")
        print(f"  Tier 1 (CrossRef):   {self.tier_stats['tier1_crossref']:,}")
        print(f"  Tier 1 (DBLP):       {self.tier_stats['tier1_dblp']:,}")
        print(f"  Tier 2 (PubMed):     {self.tier_stats['tier2_pubmed']:,}")
        print(f"  Tier 2 (EuropePMC):  {self.tier_stats['tier2_europepmc']:,}")
        print(f"  Tier 3 (DOI):        {self.tier_stats['tier3_doi']:,}")
        print(f"  Tier 4 (Text):       {self.tier_stats['tier4_text']:,}")

        print(f"\n[SUMMARY]")
        print(f"  Total recovered:     {total_recovered:,} / {len(failed_titles):,} ({total_recovered/len(failed_titles)*100:.1f}%)")
        print(f"  Final failures:      {self.tier_stats['final_failure']:,} ({self.tier_stats['final_failure']/len(failed_titles)*100:.1f}%)")
        print(f"  Time elapsed:        {elapsed/60:.1f} minutes")

        # Overall cache stats
        total_successful = sum(1 for authors in self.cache.values() if authors)
        total_papers = len(self.cache)

        print(f"\n[OVERALL SUCCESS RATE]")
        print(f"  Total papers:        {total_papers:,}")
        print(f"  With authors:        {total_successful:,} ({total_successful/total_papers*100:.1f}%)")
        print(f"  Without authors:     {total_papers - total_successful:,} ({(total_papers - total_successful)/total_papers*100:.1f}%)")

        if (total_successful/total_papers*100) >= 95:
            print(f"\n[SUCCESS] Achieved 95%+ target!")
        else:
            remaining_failures = total_papers - total_successful
            print(f"\n[INFO] {remaining_failures:,} papers still without authors")
            print(f"[RECOMMENDATION] Consider Tier 5 (manual lookup) for critical papers")


if __name__ == "__main__":
    recovery = ComprehensiveAuthorRecovery()
    recovery.process_all_failures()
