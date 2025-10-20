#!/usr/bin/env python3
"""
Production Training Script: Research Paper Classifier
Trains SciBERT on 43,517 CS research papers (8 categories)

Dataset: cs_research.db
- 43,517 papers total
- 8 balanced categories (CS subfields)
- Each category: ~5,200-5,700 papers

Target: >85% accuracy for production deployment
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import sqlite3
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ResearchPaperDataset(Dataset):
    """Dataset for research paper classification"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ResearchClassifierTrainer:
    """Production trainer for research paper classifier"""

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

        # Enable cuDNN auto-tuner for optimization
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True

        print("="*80)
        print("RESEARCH PAPER CLASSIFIER - PRODUCTION TRAINING")
        print("="*80)
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Mixed Precision: ENABLED (FP16)")
            print(f"cuDNN Benchmark: ENABLED")
        print(f"Database: {self.db_path}")
        print(f"Output: {self.output_dir}")
        print("="*80 + "\n")

    def load_data(self):
        """Load data from cs_research database"""
        print("[1/7] Loading dataset from database...")

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Load papers with title, summary, and category
        cursor.execute("""
            SELECT title, summary, category
            FROM research_papers
            WHERE title IS NOT NULL
              AND summary IS NOT NULL
              AND category IS NOT NULL
        """)

        data = cursor.fetchall()
        conn.close()

        # Prepare texts and labels
        texts = []
        labels = []
        categories = []

        for title, summary, category in data:
            text = f"{title}. {summary}"
            texts.append(text)
            if category not in categories:
                categories.append(category)
            labels.append(categories.index(category))

        self.categories = sorted(categories)  # Sort for consistency
        self.num_classes = len(self.categories)

        # Rebuild labels with sorted categories
        category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        labels = [category_to_idx[data[i][2]] for i in range(len(data))]

        print(f"   Loaded: {len(texts):,} papers")
        print(f"   Categories: {self.num_classes}")
        for i, cat in enumerate(self.categories):
            count = labels.count(i)
            print(f"     {i+1}. {cat}: {count:,} papers")

        return texts, labels

    def prepare_datasets(self, texts, labels, test_size=0.2):
        """Split and prepare datasets"""
        print(f"\n[2/7] Preparing train/validation split ({int((1-test_size)*100)}/{int(test_size*100)})...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        print(f"   Train: {len(X_train):,} samples")
        print(f"   Validation: {len(X_val):,} samples")

        # Initialize tokenizer
        print("\n[3/7] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"   Tokenizer: {self.model_name}")

        # Create datasets
        train_dataset = ResearchPaperDataset(X_train, y_train, self.tokenizer)
        val_dataset = ResearchPaperDataset(X_val, y_val, self.tokenizer)

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

    def initialize_model(self, learning_rate=2e-5):
        """Initialize model, optimizer, and scheduler"""
        print(f"\n[5/7] Initializing model...")

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Mixed precision scaler
        self.scaler = GradScaler() if self.device == 'cuda' else None

        print(f"   Model: {self.model_name}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Optimizer: AdamW")

    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)

        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)

        return avg_loss, accuracy, predictions, true_labels

    def train(self, num_epochs=3, batch_size=16):
        """Main training loop"""
        print(f"\n[6/7] Training model ({num_epochs} epochs)...")

        # Load and prepare data
        texts, labels = self.load_data()
        train_dataset, val_dataset = self.prepare_datasets(texts, labels)
        train_loader, val_loader = self.create_dataloaders(
            train_dataset, val_dataset, batch_size
        )
        self.initialize_model()

        print(f"\n{'='*80}")
        print("TRAINING START")
        print(f"{'='*80}\n")

        best_val_accuracy = 0
        training_history = []

        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)

            # Log results
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")

            # Save if best
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                model_path = self.output_dir / f'scibert_best_acc{val_acc*100:.1f}.pth'
                torch.save(self.model.state_dict(), model_path)
                print(f"  [BEST] Saved model: {model_path.name}")

            # Save history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })

        # Final evaluation
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"\nBest Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")

        # Classification report
        print("\nFinal Classification Report:")
        print(classification_report(
            val_labels,
            val_preds,
            target_names=self.categories,
            zero_division=0
        ))

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'categories': self.categories,
            'best_val_accuracy': best_val_accuracy,
            'training_history': training_history,
            'database_path': str(self.db_path),
            'timestamp': datetime.now().isoformat(),
            'num_epochs': num_epochs,
            'batch_size': batch_size
        }

        metadata_path = self.output_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[7/7] Training metadata saved: {metadata_path}")
        print(f"\n{'='*80}\n")

        return metadata

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Research Paper Classifier')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--output-dir', type=str, default='production/models/scibert', help='Output directory')

    args = parser.parse_args()

    # Run training
    trainer = ResearchClassifierTrainer(output_dir=args.output_dir)
    metadata = trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)

    # Check if meets production threshold
    if metadata['best_val_accuracy'] >= 0.85:
        print(f"[SUCCESS] Model ready for production deployment (accuracy: {metadata['best_val_accuracy']*100:.2f}%)")
    else:
        print(f"[WARNING] Model below production threshold (<85%): {metadata['best_val_accuracy']*100:.2f}%")
