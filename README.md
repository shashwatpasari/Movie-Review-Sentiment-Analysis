# Movie Review Sentiment Analysis

## Problem Statement

With the abundance of movie reviews available online, it is crucial to automatically analyze and classify the sentiment expressed in these reviews. Manual annotation is impractical at scale. This project aims to build, compare, and analyze multiple models for sentiment classification (positive/negative) on the IMDB movie reviews dataset, providing a spectrum of approaches from classical machine learning to state-of-the-art deep learning.

---

## Repository Overview

This repo provides Jupyter Notebooks for several models, each tackling IMDB sentiment analysis using different paradigms:

- **Classical ML:** Logistic Regression, SVM
- **Deep Learning:** BiLSTM+Attention, TextCNN
- **Transformers:** BERT (transfer learning)

Each notebook is fully self-contained and includes data loading, preprocessing, modeling, training, evaluation, and error analysis.

---

## Data

- **Dataset:** [Kwaai/IMDB_Sentiment](https://huggingface.co/datasets/Kwaai/IMDB_Sentiment)
- **Splits:** 25,000 train, 25,000 test
- **Labels:** Binary (0=Negative, 1=Positive)
- **Balanced:** Equal number of positive and negative reviews

---

## Models

### 1. Logistic Regression & SVM (`LR_SVM_model.ipynb`)
- **Preprocessing:** HTML removal, non-alpha filtering, advanced stopword removal (negation handling), word length features
- **Feature Engineering:** TF-IDF (unigrams, bigrams, sublinear TF, etc.)
- **Model Training:** Grid search for hyperparameters, cross-validation
- **Evaluation:** Classification report, confusion matrix, error analysis (misclassified reviews saved)

### 2. BiLSTM + Attention (`BiLSTM_model.ipynb`)
- **Preprocessing:** Lowercasing, tokenization, vocabulary building, padding/truncation (head-tail strategy)
- **Model:** Bidirectional LSTM with Attention layer (PyTorch)
- **Training:** AdamW optimizer, LR scheduler, mixed precision, validation split
- **Evaluation:** Accuracy, F1-score, confusion matrix, error analysis

### 3. TextCNN (`TextCNN_model.ipynb`)
- **Preprocessing:** Same as BiLSTM (TorchText pipeline)
- **Model:** TextCNN architecture leveraging multiple kernel sizes and max-pooling for n-gram feature extraction
- **Training:** AdamW optimizer, LR scheduler, early stopping/best model checkpointing
- **Evaluation:** Accuracy, F1-score, confusion matrix

### 4. BERT Transformer (`BERT_model.ipynb`)
- **Preprocessing:** BERT tokenizer (subword units, truncation to max length), conversion to Hugging Face Datasets
- **Model:** Pretrained `bert-base-uncased` fine-tuned for binary sequence classification
- **Training:** Only last K transformer blocks and classifier head are unfrozen (for efficiency), Hugging Face Trainer API, early stopping
- **Evaluation:** Accuracy, F1-score, confusion matrix

---

## Model Comparison

| Model                 | Accuracy (Test) | F1-Score | Highlights                                | Limitations                  |
|-----------------------|-----------------|----------|-------------------------------------------|------------------------------|
| Logistic Regression   | ~89.7%          | ~0.897   | Fast, interpretable, good baseline        | Linear, misses context       |
| SVM                   | ~89.8%          | ~0.898   | Robust, handles high-dimensional data     | Linear, slower than LR       |
| BiLSTM + Attention    | ~89.7%          | ~0.897   | Captures sequential dependencies, context | Needs more data, slower      |
| TextCNN               | ~89.3%          | ~0.893   | Efficient, strong local feature extraction| May miss long-distance context|
| BERT                  | ~91.5%          | ~0.915   | State-of-the-art, contextual, transfer learning | Computationally heavy        |

**Notes:**
- Deep learning results depend on hardware and hyperparameters; run the notebooks for your setup.
- TextCNN is often faster than RNNs, and competitive with BiLSTM for sentence-level tasks.

---

## Notebooks & Structure

```
├── LR_SVM_model.ipynb      # Logistic Regression and SVM (classical ML)
├── BiLSTM_model.ipynb      # BiLSTM + Attention (deep learning)
├── TextCNN_model.ipynb     # TextCNN (deep learning)
├── BERT_model.ipynb        # BERT (transformer, transfer learning)
├── artifacts/              # Saved models, vectorizers, error CSVs
├── data/                   # Data files (optional, mostly loaded via HuggingFace)
├── README.md               # This file
```

---

## Usage

1. **Clone the repo**
   ```bash
   git clone https://github.com/shashwatpasari/Movie-Review-Sentiment-Analysis.git
   cd Movie-Review-Sentiment-Analysis
   ```

2. **Install dependencies**
   - Recommended: Python 3.8+, use Anaconda or pip
   - Packages: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `nltk`, `torch`, `torchtext`, `torchmetrics`, `transformers`, `datasets`, `joblib`, `wordcloud`

3. **Run notebooks**
   - Open in Jupyter Notebook or Google Colab
   - Execute all cells in order for the desired model(s)

---

## Insights & Recommendations

- **Classical ML**: Fast, robust, strong baseline. Use when speed and interpretability matter.
- **TextCNN & BiLSTM**: Good for capturing sentence patterns. TextCNN is generally faster, BiLSTM can learn deeper context.
- **BERT**: For highest accuracy and state-of-the-art NLP, but requires a GPU and more resources.
- **Error analysis**: Misclassified samples are saved for further manual inspection and model improvement.

---

## Future Work

- Try more transformer architectures (RoBERTa, DistilBERT, etc.)
- Experiment with ensemble methods
- Add explainability (SHAP, LIME)
- Deploy as an API or web app
- Extend to multiclass sentiment or other domains

---

**Author:** [Shashwat Pasari](https://github.com/shashwatpasari)
