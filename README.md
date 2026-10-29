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

## Web interface (UI)

A lightweight web interface for managing the document database. It is a single static file, `index.html` (plain HTML/CSS/JavaScript, no frameworks, no build step), that talks to the FastAPI backend over HTTP.

### Features

- **Vertical list** of all documents with title, tags, creation date and last-modified date.
- **Search** by keywords and by meaning (the existing hybrid search + reranker). Documents are also matched by title, including different word endings (e.g. "сессия" finds "Расписание сессии").
- **Filters** by education level, campus and tags (multiple tags allowed).
- **Sorting** by creation date, modification date or title.
- **View** a document with its full text and source link.
- **Add**, **edit** and **delete** documents. Education level, campus and tags are chosen from drop-down lists; new tags can be entered via "Other…".

### How to run

1. Start the backend (see Setup above). It listens on `http://localhost:8000`.
2. Open `index.html` in a browser.
3. Click "настройки подключения" (connection settings) and enter the API address (default `http://localhost:8000`) and the token (`SECRET_TOKEN` from `const_variables.py`). The values are stored in the browser's `localStorage`.