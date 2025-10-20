#!/usr/bin/env python3
"""
Production Training Script: Research Paper Classifier (Multi-Label)
Trains SciBERT on research papers with proper multi-label classification

Dataset: cs_research.db
- 31,128 unique papers
- 8 categories (CS subfields) with heavy overlap
- Papers can belong to multiple categories simultaneously

Target: >85% accuracy (Hamming/Subset) for production deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import sqlite3
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, classification_report
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MultiLabelResearchPaperDataset(Dataset):
    """Dataset for multi-label research paper classification"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels  # Now multi-hot vectors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)  # Float for BCE loss
        }

class MultiLabelSciBERT(nn.Module):
    """SciBERT with multi-label classification head"""

    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class MultiLabelResearchClassifierTrainer:
    """Production trainer for multi-label research paper classifier"""

    def __init__(
        self,
        model_name='allenai/scibert_scivocab_uncased',
        database_path='/media/d1337g/SystemBackup/framework_baseline/models_denormalized/cs_research.db',
        output_dir='production/models/scibert'
    ):
        self.model_name = model_name
        self.db_path = Path(database_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True

        print("=" * 80)
        print("MULTI-LABEL RESEARCH PAPER CLASSIFIER - PRODUCTION TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Mixed Precision: ENABLED (FP16)")
            print(f"cuDNN Benchmark: ENABLED")
        print(f"Database: {self.db_path}")
        print(f"Output: {self.output_dir}")
        print(f"Classification: MULTI-LABEL (papers can have multiple categories)")
        print("=" * 80 + "\n")

    def load_data(self):
        """Load data from cs_research database - multi-label format"""
        print("[1/7] Loading dataset from database (multi-label mode)...")

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get all categories
        cursor.execute("SELECT DISTINCT category FROM research_papers ORDER BY category")
        self.categories = [row[0] for row in cursor.fetchall()]
        self.num_classes = len(self.categories)
        category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # Load papers grouped by title (to handle multi-label)
        cursor.execute("""
            SELECT title, summary, GROUP_CONCAT(category, '|') as categories
            FROM research_papers
            WHERE title IS NOT NULL
              AND summary IS NOT NULL
              AND category IS NOT NULL
            GROUP BY title, summary
        """)

        data = cursor.fetchall()
        conn.close()

        # Prepare texts and multi-hot labels
        texts = []
        labels = []

        for title, summary, categories_str in data:
            text = f"{title}. {summary}"
            texts.append(text)

            # Create multi-hot label vector
            label_vector = [0.0] * self.num_classes
            paper_categories = categories_str.split('|')
            for cat in paper_categories:
                if cat in category_to_idx:
                    label_vector[category_to_idx[cat]] = 1.0
            labels.append(label_vector)

        # Calculate category statistics
        category_counts = [0] * self.num_classes
        for label_vec in labels:
            for i, val in enumerate(label_vec):
                if val == 1.0:
                    category_counts[i] += 1

        # Calculate label cardinality and density
        labels_per_paper = [sum(label_vec) for label_vec in labels]
        avg_labels_per_paper = np.mean(labels_per_paper)
        label_density = avg_labels_per_paper / self.num_classes

        print(f"\n   Dataset Statistics:")
        print(f"   Unique papers: {len(texts):,}")
        print(f"   Categories: {self.num_classes}")
        print(f"   Avg labels per paper: {avg_labels_per_paper:.2f}")
        print(f"   Label density: {label_density*100:.1f}%")

        print(f"\n   Category Distribution:")
        for i, cat in enumerate(self.categories):
            percentage = (category_counts[i] / len(texts)) * 100
            print(f"     {i+1}. {cat}: {category_counts[i]:,} papers ({percentage:.1f}%)")

        return texts, labels

    def prepare_datasets(self, texts, labels, test_size=0.2):
        """Split and prepare datasets"""
        print(f"\n[2/7] Preparing train/validation split ({int((1-test_size)*100)}/{int(test_size*100)})...")

        # Convert labels to numpy for stratification
        labels_array = np.array(labels)

        # For stratification with multi-label, use first label as stratify key
        stratify_labels = [np.argmax(label) for label in labels]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=stratify_labels
        )

        print(f"   Train: {len(X_train):,} samples")
        print(f"   Validation: {len(X_val):,} samples")

        # Initialize tokenizer
        print("\n[3/7] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"   Tokenizer: {self.model_name}")

        # Create datasets
        train_dataset = MultiLabelResearchPaperDataset(X_train, y_train, self.tokenizer)
        val_dataset = MultiLabelResearchPaperDataset(X_val, y_val, self.tokenizer)

        return train_dataset, val_dataset

    def create_dataloaders(self, train_dataset, val_dataset, batch_size=16):
        """Create data loaders"""
        print(f"\n[4/7] Creating data loaders (batch_size={batch_size})...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )

        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")

        return train_loader, val_loader

    def initialize_model(self, learning_rate=2e-5, num_training_steps=None):
        """Initialize model, optimizer, and scheduler"""
        print(f"\n[5/7] Initializing multi-label model...")

        # Load model
        self.model = MultiLabelSciBERT(
            self.model_name,
            num_labels=self.num_classes,
            dropout=0.1
        )
        self.model.to(self.device)

        # Binary cross-entropy loss for multi-label
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Scheduler
        if num_training_steps:
            num_warmup_steps = int(0.1 * num_training_steps)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = None

        # Mixed precision scaler
        self.scaler = GradScaler() if self.device == 'cuda' else None

        print(f"   Model: {self.model_name}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Loss: BCEWithLogitsLoss (multi-label)")
        print(f"   Optimizer: AdamW")
        if self.scheduler:
            print(f"   Scheduler: Cosine with warmup")

    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler:
                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

            # Get predictions (threshold at 0.5)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        hamming = 1 - hamming_loss(all_labels, all_predictions)
        subset_acc = accuracy_score(all_labels, all_predictions)
        f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

        return avg_loss, hamming, subset_acc, f1_micro, f1_macro

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        hamming = 1 - hamming_loss(all_labels, all_predictions)
        subset_acc = accuracy_score(all_labels, all_predictions)
        f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

        return avg_loss, hamming, subset_acc, f1_micro, f1_macro, all_predictions, all_labels

    def train(self, num_epochs=10, batch_size=16):
        """Main training loop"""
        print(f"\n[6/7] Training model ({num_epochs} epochs)...")

        # Load and prepare data
        texts, labels = self.load_data()
        train_dataset, val_dataset = self.prepare_datasets(texts, labels)
        train_loader, val_loader = self.create_dataloaders(
            train_dataset, val_dataset, batch_size
        )

        num_training_steps = len(train_loader) * num_epochs
        self.initialize_model(num_training_steps=num_training_steps)

        print(f"\n{'=' * 80}")
        print("TRAINING START")
        print(f"{'=' * 80}\n")

        best_f1_macro = 0
        training_history = []

        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")

            # Train
            train_loss, train_hamming, train_subset, train_f1_micro, train_f1_macro = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_hamming, val_subset, val_f1_micro, val_f1_macro, val_preds, val_labels = self.validate(val_loader)

            # Log results
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Hamming: {train_hamming:.4f} | Subset Acc: {train_subset:.4f} | F1 Macro: {train_f1_macro:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Hamming: {val_hamming:.4f} | Subset Acc: {val_subset:.4f} | F1 Macro: {val_f1_macro:.4f}")

            # Save if best
            if val_f1_macro > best_f1_macro:
                best_f1_macro = val_f1_macro
                model_path = self.output_dir / f'scibert_multilabel_f1{val_f1_macro*100:.1f}.pth'
                torch.save(self.model.state_dict(), model_path)
                print(f"  [BEST] Saved model: {model_path.name}")

            # Save history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_hamming': train_hamming,
                'train_subset_acc': train_subset,
                'train_f1_macro': train_f1_macro,
                'val_loss': val_loss,
                'val_hamming': val_hamming,
                'val_subset_acc': val_subset,
                'val_f1_macro': val_f1_macro
            })

        # Final evaluation
        print(f"\n{'=' * 80}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 80}")
        print(f"\nBest Validation F1 Macro: {best_f1_macro:.4f} ({best_f1_macro*100:.2f}%)")

        # Per-category report
        print("\nPer-Category Performance:")
        for i, cat in enumerate(self.categories):
            cat_labels = val_labels[:, i]
            cat_preds = val_preds[:, i]
            if cat_labels.sum() > 0:
                f1 = f1_score(cat_labels, cat_preds, zero_division=0)
                precision = (cat_labels * cat_preds).sum() / (cat_preds.sum() + 1e-10)
                recall = (cat_labels * cat_preds).sum() / (cat_labels.sum() + 1e-10)
                print(f"  {cat:8s}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'categories': self.categories,
            'classification_type': 'multi-label',
            'best_f1_macro': best_f1_macro,
            'training_history': training_history,
            'database_path': str(self.db_path),
            'timestamp': datetime.now().isoformat(),
            'num_epochs': num_epochs,
            'batch_size': batch_size
        }

        metadata_path = self.output_dir / 'training_metadata_multilabel.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[7/7] Training metadata saved: {metadata_path}")
        print(f"\n{'=' * 80}\n")

        return metadata

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Multi-Label Research Paper Classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--output-dir', type=str, default='production/models/scibert', help='Output directory')

    args = parser.parse_args()

    # Run training
    trainer = MultiLabelResearchClassifierTrainer(output_dir=args.output_dir)
    metadata = trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)

    # Check if meets production threshold
    if metadata['best_f1_macro'] >= 0.85:
        print(f"[SUCCESS] Model ready for production deployment (F1: {metadata['best_f1_macro']*100:.2f}%)")
    else:
        print(f"[WARNING] Model below production threshold (<85%): {metadata['best_f1_macro']*100:.2f}%")
