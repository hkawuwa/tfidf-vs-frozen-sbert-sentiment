# TF-IDF vs. Frozen Sentence-BERT for Sentiment Classification

Code and results for the manuscript *"Sparse lexical features versus frozen sentence embeddings for
sentiment classification of long reviews: An accuracy–efficiency analysis"*
(under review).

## Overview
This study compares TF-IDF features with frozen Sentence-BERT embeddings
(all-MiniLM-L6-v2) for binary sentiment classification on the IMDb Movie
Reviews dataset. It covers:
- two preprocessing conditions (minimal and conventional cleaning)
- direct encoding vs. chunking with mean pooling for long reviews
- three classifiers (Multinomial Naïve Bayes, Logistic Regression, linear SVM)
- nested stratified 5-fold cross-validation
- timing of preprocessing, feature extraction, training and inference
- McNemar and Wilcoxon signed-rank tests, and error-overlap analysis

## Data
The IMDb Movie Reviews dataset (Maas et al., 2011) is not included in this
repository. Download it from Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Model
all-MiniLM-L6-v2, loaded from Kaggle Models (srg9000/all-minilm-l6-v2,
Transformers variant, version 1). It is used frozen (no fine-tuning).

## Repository contents
- `sentimental.ipynb`: full experiment notebook
- `results/`: fold assignments, fold-level results, pilot search results,
  cross-validation summaries and experiment configuration

## How to run
1. Open the notebook on Kaggle and turn on a GPU (an NVIDIA Tesla T4 was used).
2. Attach the IMDb dataset and the all-MiniLM-L6-v2 model listed above.
3. Run all cells. Outputs are saved to `/kaggle/working/imdb_revision/`.

## Environment
Python 3.12.13, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, NLTK,
PyTorch 2.10.0 (CUDA 12.8), Sentence-Transformers 5.4.1. Random seed: 42.

## Main result
| Configuration | Macro-F1 (mean ± SD, 5 folds) |
|---|---|
| TF-IDF (minimal cleaning) + linear SVM | 0.9069 ± 0.0032 |
| Frozen SBERT (chunked) + linear SVM | 0.8373 ± 0.0053 |

## License
Code released under the MIT License.
