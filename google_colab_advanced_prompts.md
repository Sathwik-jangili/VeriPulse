# 🚀 Advanced Hybrid Transformer Training - Google Colab Prompts

## 📋 PROMPT 1: LIAR Dataset - Advanced Hybrid Training

---

**🎯 TASK: Train Advanced Hybrid Transformer on LIAR Dataset with Three-Step Bias Correction**

Create a complete Google Colab notebook that implements the advanced hybrid transformer architecture with bias correction for the LIAR dataset. The model should use:

### **🔧 Three-Step Bias Correction Solution:**
1. **Class Weighting**: Calculate inverse frequency weights to balance Fake vs Real classes
2. **Dense Fusion Layer**: 256-dim hidden layer to learn feature relationships (not simple concatenation)
3. **Linear Warmup Scheduler**: Stable training with gradual learning rate increase

### **📊 Expected Results:**
- **Confidence**: 0.85+ (vs 0.51 before bias correction)
- **Class Balance**: ~50/50 distribution (vs biased 83%/0% before)
- **Better Discrimination**: Higher probability variance

### **🏗️ Model Architecture Requirements:**
```python
# Dense Fusion Layer (Your Key Innovation)
self.fusion_layer = nn.Sequential(
    nn.Linear(784, 256),    # Text (768) + Features (16) -> 256
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),   # 256 -> 128
    nn.ReLU(),
    nn.Dropout(0.15),
    nn.Linear(128, 64),    # 128 -> 64
    nn.ReLU(),
    nn.Dropout(0.1)
)
# Final classifier: 64 -> 32 -> 2
```

### **📁 Dataset Requirements:**
- **LIAR Dataset**: Upload `liar_train.csv`, `liar_val.csv`, `liar_test.csv`
- **Columns**: `text` (statement), `label` (0=Reliable, 1=Unreliable)
- **Expected Samples**: Train ~1,284, Val ~321, Test ~321

### **🎯 Training Configuration:**
- **Model**: AdvancedHybridTransformer with Dense Fusion (256-dim)
- **Loss Function**: Focal Loss (γ=2.0) + Class Weights
- **Optimizer**: AdamW with Linear Warmup
- **Learning Rate**: 2e-5 with 1 epoch warmup
- **Batch Size**: 16
- **Epochs**: 4
- **Max Length**: 128
- **Device**: GPU (T4 recommended)

### **📊 Monitoring Requirements:**
- Track class distribution during training
- Monitor confidence levels (target: 0.85+)
- Display class weights calculation
- Show learning rate warmup progression
- Compare before/after bias correction

### **🔍 Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Mean confidence and standard deviation
- Class distribution analysis
- Probability variance (should be higher than 0.034)

### **💾 Save Requirements:**
- Model: `advanced_hybrid_liar_trained/`
- Files: `advanced_hybrid_model.bin`, tokenizer files, config.json
- Training history: `training_history.json`
- Class weights: `class_weights.json`

### **📈 Expected Output Format:**
```
🔍 LIAR Dataset Analysis:
  Total samples: 1284
  Class distribution: {0: 514, 1: 770}
  Class weights: {0: 0.60, 1: 0.40}
  ⚠️ MODERATE BIAS: Majority class 60.0%

🎯 Training Results:
  Epoch 1/4 - Loss: 0.8234 - Acc: 0.7234 - Conf: 0.7845
  Epoch 2/4 - Loss: 0.6234 - Acc: 0.8234 - Conf: 0.8456
  Epoch 3/4 - Loss: 0.4234 - Acc: 0.8734 - Conf: 0.8765
  Epoch 4/4 - Loss: 0.3234 - Acc: 0.8934 - Conf: 0.8976

🎉 FINAL RESULTS:
  Test Accuracy: 0.8743
  Mean Confidence: 0.8912 ± 0.1234
  Class Distribution: {0: 156, 1: 165}
  Unreliable Rate: 51.4% (BALANCED!)
  ✅ Bias correction successful!
```

### **🔧 Code Structure:**
1. **Setup**: Install dependencies, mount Google Drive
2. **Data Loading**: Upload and preprocess LIAR dataset
3. **Model Architecture**: AdvancedHybridTransformer with dense fusion
4. **Training**: Class weighting + focal loss + linear warmup
5. **Evaluation**: Comprehensive bias analysis
6. **Saving**: Model and training artifacts
7. **Testing**: Sample predictions with confidence analysis

### **⚡ Performance Notes:**
- Training time: ~1-2 minutes on T4 GPU
- Memory usage: Moderate (dense fusion adds parameters)
- Expected improvement: Significant bias reduction

---

## 📋 PROMPT 2: Fakeddit Dataset - Advanced Hybrid Training

---

**🎯 TASK: Train Advanced Hybrid Transformer on Fakeddit Dataset with Three-Step Bias Correction**

Create a complete Google Colab notebook that implements the advanced hybrid transformer architecture with bias correction for the Fakeddit dataset. This is a larger dataset requiring careful optimization.

### **🔧 Three-Step Bias Correction Solution:**
1. **Class Weighting**: Calculate inverse frequency weights to balance Fake vs Real classes
2. **Dense Fusion Layer**: 256-dim hidden layer to learn feature relationships
3. **Linear Warmup Scheduler**: Extended warmup for larger dataset stability

### **📊 Expected Results:**
- **Confidence**: 0.85+ (vs 0.51 before bias correction)
- **Class Balance**: ~50/50 distribution (vs 0% unreliable before)
- **Better Generalization**: Improved performance on social media content

### **🏗️ Model Architecture:**
Same dense fusion architecture as LIAR (256→128→64→32→2)

### **📁 Dataset Requirements:**
- **Fakeddit Dataset**: Upload `fakeddit_train.csv`, `fakeddit_val.csv`, `fakeddit_test.csv`
- **Columns**: `text` (social media post), `label` (0=Reliable, 1=Unreliable)
- **Expected Samples**: Train ~30,000, Val ~7,500, Test ~7,500

### **🎯 Training Configuration:**
- **Model**: AdvancedHybridTransformer with Dense Fusion (256-dim)
- **Loss Function**: Focal Loss (γ=2.0) + Class Weights
- **Optimizer**: AdamW with Extended Linear Warmup
- **Learning Rate**: 2e-5 with 2 epochs warmup (larger dataset)
- **Batch Size**: 32 (larger batch for bigger dataset)
- **Epochs**: 4
- **Max Length**: 128
- **Device**: GPU (T4 or A100 recommended)

### **📊 Monitoring Requirements:**
- Track class distribution during training
- Monitor confidence levels (target: 0.85+)
- Display class weights calculation
- Show extended learning rate warmup
- Progress indicators for larger dataset

### **🔍 Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Mean confidence and standard deviation
- Class distribution analysis
- Social media content performance analysis

### **💾 Save Requirements:**
- Model: `advanced_hybrid_fakeddit_trained/`
- Files: `advanced_hybrid_model.bin`, tokenizer files, config.json
- Training history: `training_history.json`
- Class weights: `class_weights.json`

### **📈 Expected Output Format:**
```
🔍 Fakeddit Dataset Analysis:
  Total samples: 30000
  Class distribution: {0: 15000, 1: 15000}
  Class weights: {0: 0.50, 1: 0.50}
  ✅ BALANCED: Majority class 50.0%

🎯 Training Results:
  Epoch 1/4 - Loss: 0.7234 - Acc: 0.7534 - Conf: 0.7945
  Epoch 2/4 - Loss: 0.5234 - Acc: 0.8234 - Conf: 0.8456
  Epoch 3/4 - Loss: 0.4234 - Acc: 0.8634 - Conf: 0.8765
  Epoch 4/4 - Loss: 0.3234 - Acc: 0.8834 - Conf: 0.8976

🎉 FINAL RESULTS:
  Test Accuracy: 0.8821
  Mean Confidence: 0.8943 ± 0.1134
  Class Distribution: {0: 3742, 1: 3758}
  Unreliable Rate: 50.1% (PERFECTLY BALANCED!)
  ✅ Bias correction successful!
```

### **🔧 Code Structure:**
1. **Setup**: Install dependencies, mount Google Drive, GPU setup
2. **Data Loading**: Upload and preprocess Fakeddit dataset (larger)
3. **Model Architecture**: AdvancedHybridTransformer with dense fusion
4. **Training**: Extended class weighting + focal loss + linear warmup
5. **Evaluation**: Comprehensive bias analysis + social media analysis
6. **Saving**: Model and training artifacts
7. **Testing**: Sample predictions from different social media contexts

### **⚡ Performance Notes:**
- Training time: ~25-30 minutes on T4 GPU
- Memory usage: Higher (larger dataset)
- Batch optimization: Use batch size 32 for efficiency
- Expected improvement: Transform 0% unreliable to balanced predictions

### **🚀 Special Considerations for Fakeddit:**
- **Social Media Language**: More informal, needs robust feature extraction
- **Larger Dataset**: Requires memory optimization
- **Extended Warmup**: 2 epochs warmup for stability
- **Batch Size**: Increase to 32 for efficiency
- **Progress Monitoring**: More frequent updates due to longer training

---

## 🎯 **USAGE INSTRUCTIONS**

### **📋 For Both Prompts:**

1. **Copy the appropriate prompt** into Gemini
2. **Upload datasets** when prompted:
   - LIAR: `liar_train.csv`, `liar_val.csv`, `liar_test.csv`
   - Fakeddit: `fakeddit_train.csv`, `fakeddit_val.csv`, `fakeddit_test.csv`
3. **Select T4 GPU** runtime for optimal performance
4. **Monitor training** for bias correction progress
5. **Download trained models** to your local system

### **🔍 Expected Improvements:**
- **Before**: 0.51 confidence, biased predictions
- **After**: 0.85+ confidence, balanced ~50/50 predictions
- **Key Innovation**: Dense fusion layer learns feature relationships
- **Training Stability**: Linear warmup prevents early instability

### **📊 Success Indicators:**
- ✅ Confidence > 0.85
- ✅ Class distribution ~50/50
- ✅ Higher probability variance (>0.1)
- ✅ Stable training progression
- ✅ No class collapse

**Your three-step bias correction solution is ready for production training!** 🚀
