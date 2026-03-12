"""
Graph Neural Network — Variant-Gene Interaction Model
=======================================================
Builds a biological protein-interaction graph from STRING DB and
trains a Graph Attention Network (GAT) to predict variant pathogenicity
by propagating variant features through the protein interaction network.

Why GNN?
  - A variant's effect depends on its gene's network context.
  - A variant in a hub gene (TP53, BRCA1) behaves differently from one in
    an isolated gene; GNNs capture this topology natively.

STRING DB (free for academic use):
    https://string-db.org/cgi/download
    Human links v12: 9606.protein.links.detailed.v12.0.txt.gz

CHANGES FROM PHASE 1:
  - Was never written to disk in Phase 1 (Bug 3 fixed).
  - nx.read_gpickle / nx.write_gpickle removed in NetworkX 3.3+.
    Replaced with stdlib pickle.dump / pickle.load (Bug 6 fixed).
  - Module-level logging.basicConfig removed (Issue L).
  - from __future__ import annotations added (Issue N).

Dependencies:
    pip install torch torch-geometric requests networkx pandas
"""

from __future__ import annotations

import gzip
import io
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx

logger = logging.getLogger(__name__)

STRING_URL = (
    "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/"
    "9606.protein.links.detailed.v12.0.txt.gz"
)
STRING_NAMES_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/"
    "9606.protein.info.v12.0.txt.gz"
)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
class StringDBGraph:
    """
    Builds a NetworkX protein-interaction graph from STRING DB.

    Nodes = genes / proteins.
    Edges = interactions with combined_score >= threshold.
    The graph is cached to disk using stdlib pickle so there is no
    dependency on the NetworkX version (Bug 6: gpickle was removed in 3.3+).
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/raw/cache"),
        combined_score_threshold: int = 700,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = combined_score_threshold
        self.graph: Optional[nx.Graph] = None
        self._protein_to_gene: dict[str, str] = {}

    # ── I/O helpers ────────────────────────────────────────────────────────

    def _download_gz(self, url: str) -> pd.DataFrame:
        logger.info("Downloading %s...", url)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        raw = b"".join(resp.iter_content(chunk_size=1 << 20))
        with gzip.open(io.BytesIO(raw), "rt") as fh:
            return pd.read_csv(fh, sep=" ")

    def _save_graph(self, G: nx.Graph, path: Path) -> None:
        """Serialize a NetworkX graph with stdlib pickle (replaces gpickle)."""
        with open(path, "wb") as fh:
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_graph(self, path: Path) -> nx.Graph:
        """Deserialize a NetworkX graph with stdlib pickle (replaces gpickle)."""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ── Protein name mapping ───────────────────────────────────────────────

    def _load_protein_names(self) -> dict[str, str]:
        """Map STRING protein ID → HGNC gene symbol."""
        cache = self.cache_dir / "string_names.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            df = self._download_gz(STRING_NAMES_URL)
            df.to_parquet(cache, index=False)
        # STRING format: "#string_protein_id", "preferred_name", ...
        id_col = "#string_protein_id" if "#string_protein_id" in df.columns else df.columns[0]
        name_col = "preferred_name" if "preferred_name" in df.columns else df.columns[1]
        return dict(zip(df[id_col], df[name_col]))

    # ── Graph building ─────────────────────────────────────────────────────

    def build(self, force_refresh: bool = False) -> nx.Graph:
        """
        Load the STRING graph from cache, or download and build it fresh.

        The cache file uses a .pkl extension (not .gpickle) to make clear
        it is written with stdlib pickle, not the removed NetworkX helper.
        """
        cache_path = self.cache_dir / f"string_graph_{self.threshold}.pkl"
        if cache_path.exists() and not force_refresh:
            logger.info("Loading cached STRING graph from %s", cache_path)
            self.graph = self._load_graph(cache_path)
            return self.graph

        self._protein_to_gene = self._load_protein_names()

        cache_links = self.cache_dir / "string_links.parquet"
        if cache_links.exists() and not force_refresh:
            links_df = pd.read_parquet(cache_links)
        else:
            links_df = self._download_gz(STRING_URL)
            links_df.to_parquet(cache_links, index=False)

        logger.info("Raw interactions: %d", len(links_df))
        links_df = links_df[links_df["combined_score"] >= self.threshold]
        logger.info("After threshold=%d: %d edges.", self.threshold, len(links_df))

        G = nx.Graph()
        for _, row in links_df.iterrows():
            p1 = self._protein_to_gene.get(row["protein1"], row["protein1"])
            p2 = self._protein_to_gene.get(row["protein2"], row["protein2"])
            G.add_edge(p1, p2, weight=row["combined_score"] / 1000.0)

        self.graph = G
        self._save_graph(G, cache_path)
        logger.info(
            "STRING graph: %d nodes, %d edges. Saved to %s.",
            G.number_of_nodes(), G.number_of_edges(), cache_path,
        )
        return G

    def subgraph_for_genes(self, genes: list[str], n_hops: int = 1) -> nx.Graph:
        """Extract the n-hop neighborhood subgraph around a set of seed genes."""
        if self.graph is None:
            raise RuntimeError("Call build() before subgraph_for_genes().")
        seed_nodes = set(genes) & set(self.graph.nodes)
        if not seed_nodes:
            return nx.Graph()
        neighbors = set(seed_nodes)
        for _ in range(n_hops):
            new_neighbors: set[str] = set()
            for node in neighbors:
                new_neighbors.update(self.graph.neighbors(node))
            neighbors |= new_neighbors
        return self.graph.subgraph(neighbors).copy()


# ---------------------------------------------------------------------------
# PyG dataset construction
# ---------------------------------------------------------------------------
def build_pyg_dataset(
    variant_df: pd.DataFrame,
    graph: nx.Graph,
    node_feature_cols: list[str],
    label_col: str = "acmg_label",
) -> list[Data]:
    """
    For each labeled variant in variant_df, build a PyTorch Geometric Data
    object that represents its local protein-interaction subgraph.

    Each Data object contains:
      - x          : node feature matrix (n_nodes × n_feats + 1 indicator)
      - edge_index : COO edge list
      - y          : binary label (1 = pathogenic)
      - gene_idx   : index of the focal gene within the node list
      - variant_id : for downstream tracking
    """
    all_genes = list(graph.nodes)
    gene_index = {g: i for i, g in enumerate(all_genes)}
    n_nodes = len(all_genes)
    n_feats = len(node_feature_cols)

    # Mean-aggregate variant features per gene across the full dataset
    gene_features = np.zeros((n_nodes, n_feats), dtype=np.float32)
    for feat_idx, feat in enumerate(node_feature_cols):
        if feat not in variant_df.columns:
            continue
        grp = variant_df.groupby("gene_symbol")[feat].mean()
        for gene, val in grp.items():
            if gene in gene_index:
                gene_features[gene_index[gene], feat_idx] = float(val)

    # Build COO edge tensor from graph (both directions for undirected)
    edge_pairs = [
        [gene_index[u], gene_index[v]]
        for u, v in graph.edges()
        if u in gene_index and v in gene_index
    ]
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        # Add reverse edges for undirected message passing
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    dataset: list[Data] = []
    labeled = variant_df[variant_df[label_col].notna()]

    for _, row in labeled.iterrows():
        gene = row.get("gene_symbol")
        if gene not in gene_index:
            continue

        # Shared node feature matrix + focal-gene indicator column
        x = torch.tensor(gene_features, dtype=torch.float)
        focal_indicator = torch.zeros(n_nodes, 1)
        focal_indicator[gene_index[gene], 0] = 1.0
        x = torch.cat([x, focal_indicator], dim=-1)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([int(row[label_col])], dtype=torch.long),
            gene_idx=gene_index[gene],
            variant_id=str(row.get("variant_id", "")),
        )
        dataset.append(data)

    logger.info("Built %d PyG data objects.", len(dataset))
    return dataset


# ---------------------------------------------------------------------------
# Graph Attention Network
# ---------------------------------------------------------------------------
class VariantGAT(nn.Module):
    """
    Graph Attention Network (GAT) for variant pathogenicity prediction.

    Architecture:
      - 3 GAT layers with multi-head attention and dropout.
      - ELU activations between layers.
      - Readout via focal-node embedding (not graph-level mean pool) to
        keep predictions variant-specific.
      - Two-class head (binary cross-entropy compatible).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_channels,          hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, out_channels,  heads=1,    concat=False, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        gene_idx: torch.Tensor,
    ) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)

        focal_embeddings = x[gene_idx]  # (batch, out_channels)
        return self.classifier(focal_embeddings)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
class GNNTrainer:
    """Manages training, evaluation, and early stopping for VariantGAT."""

    def __init__(
        self,
        model: VariantGAT,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32,
        device: Optional[str] = None,
        checkpoint_path: str = "models/best_gat.pt",
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.gene_idx)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs
        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        all_proba: list[float] = []
        all_labels: list[int] = []

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.gene_idx)
            proba = F.softmax(out, dim=-1)[:, 1].cpu().numpy()
            all_proba.extend(proba.tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())

        proba_arr = np.array(all_proba)
        label_arr = np.array(all_labels)

        auc = (
            roc_auc_score(label_arr, proba_arr)
            if len(np.unique(label_arr)) > 1
            else 0.0
        )
        logits = np.stack([1 - proba_arr, proba_arr], axis=1)
        ce_loss = F.cross_entropy(
            torch.tensor(logits, dtype=torch.float),
            torch.tensor(label_arr, dtype=torch.long),
        ).item()
        return ce_loss, auc

    def fit(
        self,
        train_dataset: list[Data],
        val_dataset: list[Data],
        patience: int = 15,
    ) -> list[dict]:
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size)

        best_val_auc = 0.0
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.evaluate(val_loader)
            self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
            }
            self.history.append(record)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), self.checkpoint_path)
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %3d | Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f",
                    epoch, train_loss, val_loss, val_auc,
                )

            if no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d. Best Val AUC: %.4f", epoch, best_val_auc,
                )
                break

        return self.history

    def predict_proba(self, dataset: list[Data]) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=self.batch_size)
        all_proba: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.gene_idx)
                proba = F.softmax(out, dim=-1)[:, 1].cpu().numpy()
                all_proba.extend(proba.tolist())
        return np.array(all_proba)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def train_gnn_pipeline(
    variant_df: pd.DataFrame,
    node_feature_cols: list[str],
    string_threshold: int = 700,
    test_split: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
) -> tuple[VariantGAT, GNNTrainer, list[dict]]:
    """
    End-to-end GNN training pipeline.

    Args:
        variant_df: DataFrame with columns gene_symbol, acmg_label,
                    and all columns in node_feature_cols.
        node_feature_cols: Feature columns to use as node attributes.
        string_threshold: Minimum STRING combined_score (0–1000).
        test_split: Fraction of data reserved for validation.
        epochs: Maximum training epochs.
        batch_size: PyG DataLoader batch size.

    Returns:
        (model, trainer, training_history)
    """
    from sklearn.model_selection import train_test_split

    builder = StringDBGraph(combined_score_threshold=string_threshold)
    graph = builder.build()

    dataset = build_pyg_dataset(variant_df, graph, node_feature_cols)
    train_data, val_data = train_test_split(dataset, test_size=test_split, random_state=42)

    # +1 for the focal-gene indicator column appended in build_pyg_dataset
    in_channels = len(node_feature_cols) + 1
    model = VariantGAT(in_channels=in_channels, hidden_channels=128, heads=8)

    trainer = GNNTrainer(model, epochs=epochs, batch_size=batch_size)
    history = trainer.fit(train_data, val_data)
    return model, trainer, history
