# Google Colab Prompts for Hybrid Model Training

---

## 🎯 Prompt 1: LIAR Hybrid Model Training

Copy and paste this prompt into Gemini for Google Colab:

```
Create a Google Colab notebook for training a Hybrid Transformer Model on the LIAR dataset for fake news detection.

## Requirements:

### Dataset: LIAR Dataset
- Files needed: liar_train.csv, liar_val.csv, liar_test.csv
- Format: CSV with 'text' and 'label' columns (0=reliable, 1=unreliable)
- Content: Political statements, fact-checked claims

### Model Architecture: Hybrid Transformer
- Text Branch: distilbert-base-uncased (CLS token embeddings, 768-dim)
- Linguistic Features: 6 normalized features
  * Text length (word count, normalized)
  * Exclamation mark count (normalized)
  * Question mark count (normalized)
  * Uppercase word ratio
  * Average word length
  * High-salience keywords (breaking, shocking, secret, warning, urgent, exclusive, revealed, scandal, conspiracy, cover-up, bombshell, sensational)
- Feature Processing: Linear(6 → 16) + ReLU
- Fusion Layer: Concatenate text (768) + features (16) = 784
- Classifier Head: Linear(784 → 128) → ReLU → Dropout(0.3) → Linear(128 → 2)

### Training Configuration:
- Full training on all LIAR samples (no quick test)
- 4 epochs
- Batch size: 16
- Learning rate: 2e-5
- Max sequence length: 128
- Save best model based on validation F1 score
- Use GPU acceleration (T4 recommended)

### Structure:
1. Install dependencies (torch, transformers, datasets, scikit-learn)
2. Mount Google Drive
3. Upload/load LIAR dataset files
4. Define Hybrid Transformer Model class
5. Implement linguistic feature extraction
6. Create custom trainer for dual inputs
7. Training loop with evaluation
8. Final test evaluation
9. Save model to Google Drive
10. Download trained model

### Key Code Elements:
```python
# Linguistic Feature Extraction
def extract_linguistic_features(texts):
    # 6 normalized features
    # Return torch.Tensor of shape (batch_size, 6)

# Hybrid Model Class
class HybridTransformerModel(nn.Module):
    def __init__(self, mode="full"):
        # Text encoder: DistilBERT
        # Feature processor: Linear(6->16)
        # Classifier: MLP(784->128->2)
    
    def forward(self, input_ids, attention_mask, meta_features):
        # Extract CLS embeddings
        # Process linguistic features
        # Fusion and classification

# Custom Trainer
class HybridTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Handle dual inputs (text + features)
```

### Output Requirements:
- Training progress with metrics (accuracy, precision, recall, F1)
- Final test results
- Model saved to Google Drive: `/content/drive/MyDrive/hybrid_liar_trained/`
- Download link for trained model
- Feature analysis during training
- Sample predictions with explanations

### Important Notes:
- Use the full LIAR dataset (no subset restrictions)
- Ensure proper GPU utilization
- Include progress bars and metric tracking
- Save model in format compatible with Hugging Face
- Provide clear instructions for uploading dataset files
- Include error handling for missing dependencies
- Add visualization of training metrics

The notebook should be ready to run immediately in Google Colab with T4 GPU runtime.
```

---

## 🎯 Prompt 2: Fakeddit Hybrid Model Training (Quick Test + Full Training)

Copy and paste this prompt into Gemini for Google Colab:

```
Create a Google Colab notebook for training a Hybrid Transformer Model on the Fakeddit dataset with TWO PHASES: quick test followed by full training.

## Phase 1: Quick Test (2K samples, 1 epoch)
## Phase 2: Full Training (30K samples, 4 epochs)

### Dataset: Fakeddit Dataset
- Files needed: fakeddit_train.csv, fakeddit_val.csv, fakeddit_test.csv
- Format: CSV with 'text' and 'label' columns (0=reliable, 1=unreliable)
- Content: Social media posts, news articles, multimedia content

### Model Architecture: Hybrid Transformer (Same as LIAR)
- Text Branch: distilbert-base-uncased (CLS token embeddings, 768-dim)
- Linguistic Features: 6 normalized features
  * Text length (word count, normalized)
  * Exclamation mark count (normalized)
  * Question mark count (normalized)
  * Uppercase word ratio
  * Average word length
  * High-salience keywords (breaking, shocking, secret, warning, urgent, exclusive, revealed, scandal, conspiracy, cover-up, bombshell, sensational)
- Feature Processing: Linear(6 → 16) + ReLU
- Fusion Layer: Concatenate text (768) + features (16) = 784
- Classifier Head: Linear(784 → 128) → ReLU → Dropout(0.3) → Linear(128 → 2)

### Training Configuration:

#### Phase 1: Quick Test
- 2,000 training samples (subset)
- 1 epoch
- Batch size: 16
- Learning rate: 2e-5
- Max sequence length: 128
- Purpose: Verify model works, test pipeline

#### Phase 2: Full Training
- All Fakeddit samples (~30K)
- 4 epochs
- Batch size: 16
- Learning rate: 2e-5
- Max sequence length: 128
- Save best model based on validation F1 score
- Use GPU acceleration (T4 recommended)

### Structure:
1. Install dependencies (torch, transformers, datasets, scikit-learn)
2. Mount Google Drive
3. Upload/load Fakeddit dataset files
4. Define Hybrid Transformer Model class (same as LIAR)
5. Implement linguistic feature extraction
6. Create custom trainer for dual inputs
7. **PHASE 1**: Quick test training
8. **PHASE 2**: Full training
9. Final test evaluation
10. Save model to Google Drive
11. Download trained model

### Key Code Elements:
```python
# Configuration for two phases
QUICK_TEST_SAMPLES = 2000
QUICK_TEST_EPOCHS = 1
FULL_TRAIN_EPOCHS = 4

# Phase switching
if quick_test_mode:
    train_dataset = train_dataset.select(range(QUICK_TEST_SAMPLES))
    num_epochs = QUICK_TEST_EPOCHS
else:
    num_epochs = FULL_TRAIN_EPOCHS

# Linguistic Feature Analysis (Fakeddit specific)
def analyze_fakeddit_features():
    # Analyze salience keywords in social media content
    # Compare punctuation patterns vs LIAR dataset
    # Show feature statistics

# Hybrid Model (reuse same architecture)
class HybridTransformerModel(nn.Module):
    # Same implementation as LIAR
```

### Output Requirements:
#### Phase 1 Output:
- Quick test results (verify model works)
- Training time for quick test
- Sample predictions with feature analysis
- Confirmation to proceed to full training

#### Phase 2 Output:
- Full training progress with metrics
- Feature analysis specific to Fakeddit dataset
- Comparison with quick test results
- Final test results
- Model saved to Google Drive: `/content/drive/MyDrive/hybrid_fakeddit_trained/`
- Download link for trained model

### Special Requirements:
1. **Two-Phase Training**: Clear separation between quick test and full training
2. **Fakeddit Feature Analysis**: Show how linguistic features differ from LIAR
3. **Progress Comparison**: Compare quick test vs full training performance
4. **Dataset Statistics**: Show Fakeddit dataset characteristics
5. **Social Media Focus**: Highlight differences from political LIAR dataset

### Important Notes:
- Start with quick test to verify everything works
- Allow user to confirm before proceeding to full training
- Fakeddit has different patterns than LIAR (more sensational content)
- Include feature comparison between quick test and full training
- Save both quick test and full models if possible
- Provide clear instructions for uploading dataset files
- Include error handling and GPU optimization

The notebook should guide users through both phases with clear checkpoints and results comparison.
```

---

## 📋 Usage Instructions:

1. **For LIAR Training**: Use Prompt 1
2. **For Fakeddit Training**: Use Prompt 2 (includes quick test + full training)
3. **Upload Datasets**: Make sure to upload the respective CSV files to Colab
4. **GPU Runtime**: Use T4 GPU for optimal performance
5. **Google Drive**: Models will be saved to your Google Drive

## 🎯 Expected Outcomes:

- **LIAR Model**: Specialized for political statements, fact-checking
- **Fakeddit Model**: Specialized for social media content, news articles
- **Feature Analysis**: Domain-specific linguistic patterns
- **Performance Metrics**: Accuracy, Precision, Recall, F1 scores
- **Downloadable Models**: Ready for local inference
