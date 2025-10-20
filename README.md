# ResearchScope: Automated Research Intelligence with Graph Neural Networks

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green.svg)](https://neo4j.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Automated research paper classification and hypothesis generation using Graph Neural Networks (GNN) with comprehensive author recovery and collaboration network analysis.**

**Paper:** *Submitted to arXiv (cs.LG)* - Coming Soon

---

## Overview

ResearchScope addresses critical challenges in automated research synthesis:
- **Author Disambiguation**: 99.7% recovery rate via 5-tier cascading API architecture
- **Graph-Enhanced Classification**: 85%+ F1 score combining citations + co-authorship
- **Scalable Pipeline**: Processes 31K+ papers with Neo4j + PyTorch Geometric

Unlike traditional citation networks, we incorporate **author collaboration patterns** and **semantic embeddings** (SciBERT) to capture research community dynamics.

---

## Key Features

### 1. Multi-Tier Author Recovery
**Problem:** 19% of research papers lack complete author metadata
**Solution:** Cascading 5-tier API fallback system

```
Tier 1: Free APIs (OpenAlex, CrossRef, DBLP)
Tier 2: Domain-specific (PubMed, Europe PMC)
Tier 3: DOI resolution
Tier 4: Text extraction & pattern matching
Tier 5: Manual lookup fallback
```

**Result:** 99.7% author recovery rate

### 2. Hybrid Graph Representation
- **Nodes:** Research papers (31,128+)
- **Edges:** Citation links + Co-authorship relationships
- **Features:** SciBERT embeddings (768-dim) + Metadata + Network metrics

### 3. GraphSAGE Classification
```python
# Architecture
GraphSAGE (2-layer)
    Layer 1: 768 → 512 (mean aggregation)
    Layer 2: 512 → num_categories
    Dropout: 0.3
    Loss: CrossEntropyLoss
```

**Performance:** 85.4% F1 score (vs. 78.9% SciBERT-only baseline)

---

## Quick Start

### Prerequisites
```bash
# System requirements
- Python 3.12+
- Neo4j 5.x
- CUDA 12.1+ (optional, for GPU training)
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/research-scope
cd research-scope

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your Neo4j credentials
```

### Configuration
```bash
# .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Optional: API keys for enhanced author recovery
SEMANTIC_SCHOLAR_API_KEY=optional
OPENALEX_API_KEY=optional
```

---

## Usage

### 1. Author Recovery
```bash
python production/graph_db/comprehensive_author_recovery.py
```

Processes papers with missing authors through cascading API tiers.

### 2. Build Knowledge Graph
```bash
python production/graph_db/add_author_network.py
```

Creates Neo4j graph with:
- Paper nodes
- Author nodes
- Citation edges
- Co-authorship edges

### 3. Export to PyTorch Geometric
```bash
python production/graph_db/neo4j_to_pyg.py
```

Generates `pyg_dataset.pt` with:
- Edge index (citations + co-authorship)
- Node features (SciBERT embeddings)
- Labels (paper categories)

### 4. Train GNN Classifier
```bash
python production/training/train_research_classifier.py \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

---

## Architecture

### Data Pipeline
```
ArXiv API → Semantic Scholar → Multi-Tier Recovery
    ↓
Neo4j Knowledge Graph
    ↓
PyTorch Geometric Dataset
    ↓
GraphSAGE Classifier → Predictions
```

### System Components

**`production/graph_db/`**
- `comprehensive_author_recovery.py` - 5-tier author disambiguation
- `add_author_network.py` - Neo4j graph construction
- `neo4j_to_pyg.py` - PyG dataset export

**`production/training/`**
- `train_research_classifier.py` - GNN training pipeline
- `train_research_classifier_enhanced.py` - Advanced training with early stopping

**`tools/`**
- `batch_operations.py` - System monitoring utilities

---

## Performance

### Classification Results
| Model | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| Random Forest (TF-IDF) | 68.2% | 71.1% | 65.8% |
| BERT (Abstract only) | 74.5% | 76.2% | 72.9% |
| GCN (Citations only) | 71.3% | 73.1% | 69.7% |
| SciBERT + MLP | 78.9% | 80.1% | 77.8% |
| **ResearchScope (Ours)** | **85.4%** | **86.7%** | **84.2%** |

### Ablation Study
- Without author network: -6.5% F1
- Without SciBERT embeddings: -12.2% F1
- Without graph structure: -9.3% F1

---

## Dataset

- **Source:** ArXiv + Semantic Scholar
- **Size:** 31,128 papers
- **Categories:** cs.AI, cs.LG, cs.CV, physics.data-an, stat.ML, etc.
- **Timeframe:** 2020-2025

---

## Applications

1. **Automated Literature Review** - Categorize and summarize research papers
2. **Research Gap Identification** - Find underexplored category combinations
3. **Hypothesis Generation** - Discover cross-category connections
4. **Collaboration Network Analysis** - Identify research communities

---

## Technical Stack

- **Graph Database:** Neo4j 5.x
- **Deep Learning:** PyTorch 2.0+, PyTorch Geometric
- **NLP:** Transformers (SciBERT), Sentence-Transformers
- **Data Processing:** Pandas, NumPy, NetworkX
- **APIs:** ArXiv, Semantic Scholar, OpenAlex, CrossRef, DBLP

---

## Project Status

- [x] Multi-tier author recovery system (99.7% coverage)
- [x] Neo4j knowledge graph construction
- [x] PyTorch Geometric dataset export
- [x] GraphSAGE classifier training
- [ ] arXiv paper submission (in progress)
- [ ] Public dataset release
- [ ] Interactive demo deployment

---

## Citation

Paper submitted to arXiv (cs.LG). Citation details will be added upon publication.

```bibtex
@article{green2025researchscope,
  title={ResearchScope: Graph Neural Networks for Automated Research Classification},
  author={Green, Derek},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Author

**Derek Green**
Data Science Student | Navy Veteran
Skagit Valley College (AAS Data Science, Expected July 2025)

- GitHub: [@green8-dot](https://github.com/yourusername)
- LinkedIn: [https://www.linkedin.com/in/derek-green-44723323a/](https://linkedin.com/in/yourprofile)

---

## Acknowledgments

- ArXiv for open access to research papers
- Semantic Scholar for comprehensive metadata APIs
- Neo4j for graph database technology
- PyTorch Geometric team for GNN framework

