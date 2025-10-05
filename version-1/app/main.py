import os
import logging
import traceback
import uvicorn
import pandas as pd
import json
from fastapi import FastAPI, HTTPException, Header, Request, Response
from pydantic import BaseModel
import fitz
from docx import Document
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import Optional, List

# Импорт констант (предполагается, что const_variables.py существует)
from const_variables import SECRET_TOKEN, ELASTICSEARCH_URL, INDEX_NAME, DOCS_DIR, META_PATH

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация модели и Elasticsearch клиента
model = None
es_client = None

# FastAPI приложение
app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    edu_level: Optional[str] = None
    campus: Optional[str] = None
    topic_tag: Optional[List[str]] = None
    bm25_weight: float = 0.5
    embed_weight: float = 0.5
    word_document_boost_weight: float = 1.0
    fragment_size: int = 200
    num_fragment_per_doc: int = 1
    count_doc_return: int = 10
    count_doc_for_rescore: int = 50

def get_es_client(token: str):
    global es_client
    if es_client is not None:
        return es_client

    logger.info("Подключение к Elasticsearch...")
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        es_client = Elasticsearch(
            ELASTICSEARCH_URL,
            headers={"Authorization": f"Bearer {token}"}
        )
        if not es_client.ping():
            raise ConnectionError("Connection to Elasticsearch failed")
        logger.info("Успешное подключение к Elasticsearch")
        return es_client
    except Exception as e:
        logger.error(f"Ошибка подключения: {e}")
        raise HTTPException(status_code=500, detail="Elasticsearch connection error")


def create_index(es: Elasticsearch):
    if not es.indices.exists(index=INDEX_NAME):
        try:
            es.indices.create(
                index=INDEX_NAME,
                body={
                    "mappings": {
                        "properties": {
                            "file_name": {"type": "text"},
                            "text": {"type": "text"},
                            "embedding": {"type": "dense_vector", "dims": 1024},
                            "edu_level": {"type": "keyword"},
                            "campus": {"type": "keyword"},
                            "topic_tag": {"type": "keyword"},
                            "url": {"type": "text"}
                        }
                    }
                }
            )
            logger.info(f"Индекс {INDEX_NAME} создан")
        except Exception as e:
            logger.error(f"Ошибка создания индекса: {e}")
            raise


def load_model():
    global model
    if model is None:
        logger.info("Загрузка модели SentenceTransformer...")
        model = SentenceTransformer('intfloat/multilingual-e5-large')
    return model


def get_embedding(text: str):
    if model is None:
        load_model()
    embedding = model.encode(text, convert_to_numpy=True)

    # Reshape the embedding to ensure it is a 2D array
    return normalize(embedding.reshape(1, -1))[0].tolist()  # Reshaping to 1 x N (2D array)

def read_pdf(file_path: str) -> str:
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc):
                page_text = page.get_text()
                logger.debug(f"PDF: прочитана страница {i + 1}, длина текста: {len(page_text)} символов")
                text += page_text
        return text
    except Exception as e:
        logger.error(f"Ошибка чтения PDF файла {file_path}: {e}")
        logger.debug(traceback.format_exc())
        return ""


def read_document(file_path) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    logger.debug(f"Обработка файла {file_path}, расширение: {ext}")

    if ext == ".docx":
        try:
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            logger.debug(f"DOCX: прочитано параграфов: {len(paragraphs)}")
            return "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Ошибка чтения DOCX файла {file_path}: {e}")
            logger.debug(traceback.format_exc())
            return ""
    elif ext == ".pdf":
        return read_pdf(file_path)
    else:
        logger.warning(f"Неподдерживаемый формат файла: {file_path}")
        return ""


def index_document(es: Elasticsearch, document: dict):
    try:
        logger.debug(f"Отправка документа в индекс. file_name: {document.get('file_name')}, "
                     f"embedding: {type(document.get('embedding'))}, длина embedding: {len(document.get('embedding', []))}")
        es.index(index=INDEX_NAME, document=document)
    except Exception as e:
        logger.error(f"Ошибка индексации документа {document.get('file_name')}: {e}")
        logger.debug("Содержимое документа для отладки:")
        logger.debug(json.dumps(document, indent=2, ensure_ascii=False))
        logger.debug(traceback.format_exc())
        raise


def index_documents_from_directory(es: Elasticsearch):
    try:
        meta_df = pd.read_excel(META_PATH)
        required_columns = ["file_name", "edu_level", "campus", "topic_tag", "url"]
        if not all(col in meta_df.columns for col in required_columns):
            raise ValueError(f"Отсутствуют необходимые колонки: {required_columns}")
        logger.debug(f"Загружено строк из метаданных: {len(meta_df)}")
    except Exception as e:
        logger.error(f"Ошибка чтения метаданных: {e}")
        logger.debug(traceback.format_exc())
        raise

    indexed_count = 0
    for _, row in meta_df.iterrows():
        try:
            file_path = os.path.join(DOCS_DIR, row["file_name"])
            logger.debug(f"Обработка файла: {file_path}")

            if not os.path.exists(file_path):
                logger.warning(f"Файл не найден: {file_path}")
                continue

            text = read_document(file_path)
            if not text:
                logger.warning(f"Файл пустой или не прочитан: {file_path}")
                continue

            # Обработка тегов
            if isinstance(row["topic_tag"], str):
                raw_tag = row["topic_tag"].strip()
                # Замена одинарных кавычек на двойные для корректного парсинга JSON
                raw_tag = raw_tag.replace("'", '"')
                try:
                    tags = json.loads(raw_tag)
                except json.JSONDecodeError:
                    # Если не JSON, разделяем по запятым
                    tags = [tag.strip() for tag in raw_tag.split(",")]
            else:
                tags = []

            # Удаление пустых тегов
            tags = [tag for tag in tags if tag]

            document = {
                "file_name": row["file_name"],
                "text": text,
                "edu_level": [s.strip() for s in str(row["edu_level"]).split(",")],
                "campus": [s.strip() for s in str(row["campus"]).split(",")],
                "topic_tag": tags,
                "url": row["url"] if pd.notna(row["url"]) else "",
                "embedding": get_embedding(text)
            }

            index_document(es, document)
            indexed_count += 1
        except Exception as e:
            logger.error(f"Ошибка обработки файла {row.get('file_name')}: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"Проиндексировано документов: {indexed_count}/{len(meta_df)}")
    return indexed_count

@app.on_event("startup")
async def initialize_app():
    try:
        logger.info("Инициализация приложения...")

        # Подключение к Elasticsearch
        es = get_es_client(SECRET_TOKEN)
        create_index(es)

        # Загрузка модели
        load_model()

        # Индексация документов
        index_documents_from_directory(es)

        logger.info("Инициализация завершена успешно")
    except Exception as e:
        logger.error(f"Фатальная ошибка инициализации: {e}")
        raise RuntimeError("Application initialization failed")


@app.get("/")
def health_check():
    return {"status": "OK", "message": "Document Search API"}


@app.post("/search")
def search_documents(request: SearchRequest, token: str = Header(...)):
    try:
        es = get_es_client(token)
        query_embedding = get_embedding(request.query)

        # Построение фильтров
        filters = []
        if request.edu_level:
            filters.append({"term": {"edu_level": request.edu_level}})
        if request.campus:
            filters.append({"term": {"campus": request.campus}})
        if request.topic_tag:
            filters.append({"terms": {"topic_tag": request.topic_tag}})

        search_body = {
            "_source": ["file_name", "url", "topic_tag", "text"],
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": filters,
                            "should": [
                                {
                                    "match": {
                                        "text": {
                                            "query": request.query,
                                            "boost": request.word_document_boost_weight
                                        }
                                    }
                                }
                            ],
                            "minimum_should_match": 0
                        }
                    },
                    "script": {
                        "source": """
                            double bm25 = _score;
                            double norm_bm25 = bm25 / (bm25 + 1.0);

                            double cosine = cosineSimilarity(params.query_vector, 'embedding');
                            double norm_cosine = (cosine + 1.0) / 2.0;

                            return (params.bm25_weight * norm_bm25) + (params.embed_weight * norm_cosine);
                        """,
                        "params": {
                            "query_vector": query_embedding,
                            "bm25_weight": request.bm25_weight,
                            "embed_weight": request.embed_weight
                        }
                    }
                }
            },
            "highlight": {
                "fields": {
                    "text": {
                        "fragment_size": request.fragment_size,
                        "number_of_fragments": request.num_fragment_per_doc,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            },
            "size": request.count_doc_return
        }
        response = es.search(index=INDEX_NAME, body=search_body)
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "score": round(hit["_score"], 4),
                "file_name": hit["_source"].get("file_name"),
                "url": hit["_source"].get("url"),
                "tags": hit["_source"].get("topic_tag"),
                "context": hit.get("highlight", {}).get("text", [""])[0]
            })
        return results

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)