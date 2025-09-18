# CharacterChannelEmbedding

This repository contains the implementation of `CharacterChannelEmbedding`, a novel text embedding technique designed to enhance search algorithms. It captures fine-grained text features, including QWERTY keyboard proximity, to improve search relevance for applications like YouTube, Instagram and other search platforms. The approach is lightweight, flexible, and robust to typos and variations.

## What It Is

`CharacterChannelEmbedding` is a text embedding method that transforms text into a multi-channel representation, capturing character-level and n-gram features. Unlike traditional embeddings like word2vec or BERT, it focuses on low-level text properties such as frequency, case, position, and QWERTY keyboard proximity, making it ideal for search algorithms where typo tolerance and fine-grained similarity are critical. It can replace or enhance embeddings used in search engines, video platforms, or e-commerce systems to improve query-document matching.

## How It Works

The embedding process converts text into a dictionary of tokens (characters or n-grams), each associated with a 9-dimensional feature vector (channels). These channels encode various text properties, and the resulting embeddings are normalized and weighted for similarity computation. Key components include:

- **Token Representation**: Each character or n-gram is encoded as a feature vector, capturing properties like frequency, case, and position.
- **Similarity Computation**: Uses a combination of channel-based cosine similarity (via linear sum assignment) and Levenshtein distance to measure text similarity, weighted at 85% and 15%, respectively.
- **Visualization**: Provides heatmap visualizations of embeddings and PCA projections for exploring the embedding space.

### Highlight: QWERTY Proximity

A unique feature is the incorporation of QWERTY keyboard proximity, which enhances typo tolerance in search queries. The method maps characters to their positions on a QWERTY keyboard and calculates Euclidean distances between them. For example, 'q' and 'w' are close on the keyboard, so the embedding assigns higher similarity to typos. This is particularly useful for search engines where users may mistype queries, improving results for platforms like YouTube or Google.

## Why Channels?

The multi-channel design (9 by default) is crucial for capturing diverse text properties, making the embedding robust for search applications:

- **Frequency**: Counts how often a token appears, emphasizing important characters or n-grams.
- **Case**: Differentiates between uppercase and lowercase, capturing stylistic variations.
- **Position**: Encodes where tokens appear in the text, supporting positional relevance in search.
- **Character Type**: Distinguishes letters, digits, punctuation, and spaces, aiding in understanding text structure.
- **Word Boundary Density**: Identifies tokens at word boundaries, enhancing phrase detection.
- **Local Context**: Captures neighboring character information, improving contextual understanding.
- **Phonetic Features**: Differentiates vowels and consonants, supporting phonetic similarity.
- **QWERTY X/Y Coordinates**: Encodes keyboard positions, enabling typo-aware similarity.

This multi-faceted approach ensures the embedding captures both syntactic and some semantic nuances, making it suitable for replacing or augmenting embeddings in search systems.

## Use Cases

- **Search Engines**: Enhances query-document matching for platforms like Google by handling typos and partial matches effectively.
- **Video Platforms**: Improves search on YouTube by matching queries to video titles/descriptions with typo tolerance.
- **E-commerce**: Boosts product search by aligning user queries with product names, even with misspellings.
- **Autocomplete Systems**: Supports real-time query suggestions by ranking similar terms based on embedding similarity.

## Getting Started

To use the embedding, initialize the `CharacterChannelEmbedding` class and leverage its methods:

```python
from character_channel_embedding import CharacterChannelEmbedding

embedder = CharacterChannelEmbedding(use_ngrams=True, n=3)

embedding = embedder.create_embedding("hello world")

sim = embedder.calculate_similarity("hello", "jello")

query = "brown fox jumps"
documents = ["The quick brown fox jumps over the lazy dog", ...]
results = embedder.search(query, documents, top_k=5)
```
