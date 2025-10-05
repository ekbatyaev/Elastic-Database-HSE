# Elastic-Database-HSE

**Elastic-Database-HSE** is a **vector-based document database** built on **Elasticsearch**, integrating **embedding** and **cross-encoder** models for semantic search and reranking.

## Overview

- Uses **embedding models** to convert documents and queries into dense vector representations for **semantic similarity search** in Elasticsearch.  
- Applies a **cross-encoder model** to **rerank** top results, improving precision and relevance.
- **Smart Splitter** – advanced text chunking with semantic awareness for better embedding quality. 
- Combines **fast vector retrieval** with **accurate deep reranking**, suitable for intelligent document and knowledge-base search.
  
## 📂 Structure

<img width="520" height="214" alt="изображение" src="https://github.com/user-attachments/assets/5ab21e56-6770-47fc-a002-821ebce456a3" />


## Splitter Main Features

- Splits long documents into **semantically meaningful chunks**.  
- Supports **semantic weighting**, **overlapping sentences**, and **language-specific rules** (`ru` / `en`).  
- Handles **lists**, **numbered items**, and **abbreviations** intelligently.  
- Falls back to a **simple or semantic split** for edge cases or very long texts.

## Setup

```bash
git clone https://github.com/ekbatyaev/Elastic-Database-HSE.git
cd Elastic-Database-HSE/version-3-actual
pip install -r requirements.txt

```

