# CharacterChannelEmbedding

This repository contains `CharacterChannelEmbedding`, a novel text embedding technique designed to enhance search algorithms for platforms like YouTube, Instagram, and other search systems. It captures fine-grained text features, including QWERTY keyboard proximity, and integrates convolutional neural network operations for advanced search, similarity computation, and dimensionality reduction. The approach is lightweight, typo-tolerant, and ideal for improving query-document matching.

## What It Is

`CharacterChannelEmbedding` transforms text into a multi-channel representation, capturing character-level and n-gram features such as frequency, case, position, and QWERTY keyboard proximity. Unlike traditional embeddings like word2vec or BERT, it focuses on low-level text properties, making it ideal for search algorithms where typo tolerance and fine-grained similarity are critical. Enhanced with CNN operations, it extracts higher-level patterns, enabling robust search, similarity, and visualization capabilities for applications like video platforms, e-commerce, and autocomplete systems.

## How It Works

The embedding process converts text into a dictionary of tokens (characters or n-grams), each associated with a 9-dimensional feature vector (channels). These channels encode text properties, and the resulting embeddings are normalized and weighted for similarity computation. CNN operations further process these embeddings to capture local patterns, enhancing performance for search-related tasks.

Key components include:
- **Token Representation**: Encodes characters or n-grams into feature vectors capturing frequency, case, position, and more.
- **Similarity Computation**: Combines channel-based cosine similarity and Levenshtein distance. CNN-based similarity uses convolutional layers for richer feature extraction.
- **CNN Operations**: A 1D CNN processes the embedding matrix to produce fixed-size vectors for search, similarity, and dimensionality reduction.
- **Visualization**: Provides heatmap visualizations and PCA projections of the embedding space, including CNN-based reduced representations.

### Highlight: QWERTY Proximity

A unique feature is QWERTY keyboard proximity, which enhances typo tolerance in search queries. Characters are mapped to their positions on a QWERTY keyboard, and Euclidean distances are calculated to assign higher similarity to typos like "xat" vs. "cat" (since 'x' and 'c' are close). This is crucial for search engines like YouTube or Instagram, where users often mistype queries, improving result relevance.

## Why Channels?

The multi-channel design (9 channels by default) captures diverse text properties, making the embedding robust for search applications:
- **Frequency**: Counts token occurrences, emphasizing important characters or n-grams.
- **Case**: Differentiates uppercase/lowercase, capturing stylistic variations.
- **Position**: Encodes token positions, supporting positional relevance in search.
- **Character Type**: Distinguishes letters, digits, punctuation, and spaces for text structure.
- **Word Boundary Density**: Identifies tokens at word boundaries, enhancing phrase detection.
- **Local Context**: Captures neighboring character information for contextual understanding.
- **Phonetic Features**: Differentiates vowels/consonants, supporting phonetic similarity.
- **QWERTY X/Y Coordinates**: Encodes keyboard positions for typo-aware similarity.

The CNN operations build on these channels by learning local patterns (e.g., n-gram interactions), improving robustness for search, similarity, and visualization tasks.

## Features

- **Typo Tolerance**: QWERTY proximity ensures robustness to typing errors.
- **N-gram Support**: Captures multi-character patterns for richer embeddings.
- **CNN Enhancements**: Extracts higher-level features for search, similarity, and visualization.
- **Flexible Training**: Supports supervised (with labeled similarity scores) and unsupervised training for CNN-based similarity.
- **Visualization Tools**: Includes heatmap visualizations and PCA projections for both raw and CNN-processed embeddings.
- **Lightweight**: No heavy pre-trained models required, unlike BERT-based embeddings.

## CNN Enhancements

The embedding is enhanced with a 1D CNN to process the multi-channel representation, enabling:
- **Search**: Encodes queries and documents into fixed-size vectors, ranking results by cosine similarity for improved relevance.
- **Similarity**: Computes similarity using CNN-encoded vectors, supporting both supervised training (with labeled similarity scores) and unsupervised training (with predefined pairs).
- **Dimensionality Reduction**: Uses CNN outputs as compressed representations, combined with PCA for visualization or downstream tasks.

These enhancements make the embedding suitable for replacing or augmenting traditional embeddings in search systems, capturing complex patterns while maintaining typo tolerance.

## Use Cases

- **Search Engines**: Enhances query-document matching for platforms like Google by handling typos and partial matches.
- **Video Platforms**: Improves YouTube search by matching queries to video titles/descriptions with typo tolerance.
- **Social Media**: Boosts Instagram search for hashtags, captions, or usernames, accommodating misspellings.
- **E-commerce**: Aligns user queries with product names, even with typos, for better product discovery.
- **Autocomplete Systems**: Ranks similar terms for real-time query suggestions using CNN-based similarity.

## Getting Started

To use the embedding, initialize the `CharacterChannelEmbedding` class and leverage its methods, including CNN operations:

```python
from chan_embedding import CharacterChannelEmbedding
import pandas as pd
import matplotlib.pyplot as plt

# Initialize
embedder = CharacterChannelEmbedding(use_ngrams=True, n=3)

# Create embedding
embedding = embedder.create_embedding("hello world")

# Basic similarity
sim = embedder.calculate_similarity("hello", "jello")

# Basic search
query = "brown fox jumps"
documents = ["The quick brown fox jumps over the lazy dog", ...]
results = embedder.search(query, documents, top_k=5)

# CNN-based operations
# 1. Train CNN (Supervised)
df = pd.DataFrame({
    'text1': ['cat', 'hello', 'quick', 'test', 'chopper'],
    'text2': ['bat', 'jello', 'quack', 'testing', 'hopper'],
    'similarity': [0.9, 0.8, 0.7, 0.85, 0.75]
})
embedder.train_cnn_similarity(df=df, epochs=5, save_path="cnn_model.pth")

# 2. CNN-Based Search
print("=== CNN-Based Search ===")
query = "brown fox jumps"
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps above a sleepy canine",
    "The slow red cat walks under the energetic bird",
    "Quick brown foxes jump over lazy dogs",
    "Python is a programming language"
]
results = embedder.cnn_search(query, documents, top_k=3, model_path="cnn_model.pth")
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f}, Doc: {doc}")

# 3. CNN-Based Similarity
print("\n=== CNN-Based Similarity ===")
similarity_tests = [
    ("cat", "bat"), ("hello", "jello"), ("quick", "quack"),
    ("test", "testing"), ("chopper", "hopper"), ("dog", "dogg"),
    ("programming", "programing"), ("youtube", "youtub"),
    ("machine learning", "AI learning"), ("fast car", "quick vehicle"),
    ("apple", "aple"), ("search engine", "search motor"),
    ("the quick fox", "quick fox"), ("data science", "data analysis"),
    ("python code", "python script")
]
for text1, text2 in similarity_tests:
    sim = embedder.cnn_similarity(text1, text2, model_path="cnn_model.pth")
    print(f"CNN Similarity('{text1}', '{text2}'): {sim:.4f}")

# 4. CNN-Based Dimensionality Reduction
print("\n=== CNN-Based Dimensionality Reduction ===")
words = ["cat", "bat", "hat", "dog", "fog", "quick", "brown"]
reduced, variance = embedder.cnn_reduce_dimensionality(words, target_dim=2, model_path="cnn_model.pth")
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], s=100, alpha=0.7)
for i, word in enumerate(words):
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]), xytext=(5, 5), textcoords="offset points")
plt.title("CNN + PCA Embedding Space")
plt.xlabel(f"PC1 ({variance[0]:.1%} variance)")
plt.ylabel(f"PC2 ({variance[1]:.1%} variance)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"PCA Explained Variance: PC1={variance[0]:.1%}, PC2={variance[1]:.1%}")

# 5. Train CNN (Unsupervised)
print("\n=== Training CNN (Unsupervised) ===")
embedder.train_cnn_similarity(epochs=5, save_path="cnn_model_unsupervised.pth")
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.