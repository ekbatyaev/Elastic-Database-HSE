import os
import logging
import traceback
from datetime import datetime
from datetime import timezone, timedelta
from pathlib import Path
import re
from typing import Tuple
import fitz
import uvicorn
import pandas as pd
import json
import numpy as np

from elasticsearch.helpers import bulk
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from docx import Document
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
from typing import Optional, List
import hashlib
import spacy

# Импорт констант
from const_variables import SECRET_TOKEN, ELASTICSEARCH_URL, CHUNK_INDEX, PARENT_INDEX, DOCS_DIR, META_PATH, \
    ACTUAL_LOGS_PATH

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
reranker = None
es_client = None
nlp = spacy.load("ru_core_news_sm")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    edu_level: Optional[str] = None
    campus: Optional[str] = None
    topic_tag: Optional[List[str]] = None
    count_doc_return: int = 10
    fragment_size: int = 200
    number_fragments_in_chunk: int = 1
    count_doc_rerank: int = 10

class AddRequest(BaseModel):
    doc_id: str
    file_name: str
    edu_level: list[str]
    campus: list[str]
    topic_tag: list[str]
    url: str
    full_text: str
    doc_source: str = ""

class DeleteRequest(BaseModel):
    file_name: str

class GetLogs(BaseModel):
    log_name: str



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
            headers={"Authorization": f"Bearer {token}"}, timeout=60
        )
        if not es_client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")
        logger.info("Успешное подключение к Elasticsearch")
        return es_client
    except Exception as e:
        logger.error(f"Ошибка подключения к Elasticsearch: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Elasticsearch connection error")


def create_indices(es: Elasticsearch):
    # Создание индекса для родительских документов
    if not es.indices.exists(index=PARENT_INDEX):
        try:
            es.indices.create(
                index=PARENT_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "file_name": {"type": "text"},
                            "edu_level": {"type": "keyword"},
                            "campus": {"type": "keyword"},
                            "topic_tag": {"type": "keyword"},
                            "url": {"type": "text"},
                            "full_text": {"type": "text"},
                            "doc_source": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"}
                        }
                    }
                }
            )
            logger.info(f"Индекс {PARENT_INDEX} создан")
        except Exception as e:
            logger.error(f"Ошибка создания индекса {PARENT_INDEX}: {e}")
            logger.debug(traceback.format_exc())
            raise

    # Создание индекса для чанков
    if not es.indices.exists(index=CHUNK_INDEX):
        try:
            es.indices.create(
                index=CHUNK_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "text": {"type": "text"},
                            "embedding": {"type": "dense_vector", "dims": 1024},
                            "topic_tag": {"type": "keyword"},
                            "edu_level": {"type": "keyword"},
                            "campus": {"type": "keyword"}
                        }
                    }
                }
            )
            logger.info(f"Индекс {CHUNK_INDEX} создан")
        except Exception as e:
            logger.error(f"Ошибка создания индекса {CHUNK_INDEX}: {e}")
            logger.debug(traceback.format_exc())
            raise


def load_model():
    global model, reranker
    if model is None:
        logger.info("Загрузка модели SentenceTransformer...")
        model = SentenceTransformer('intfloat/multilingual-e5-large')
    if reranker is None:
        logger.info("Загрузка модели CrossEncoder...")
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return model, reranker


def get_embedding(text: str):
    if model is None:
        load_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return normalize(embedding.reshape(1, -1))[0].tolist()


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


def is_potential_sentence_break(text: str) -> bool:
    """Проверяет, является ли позиция потенциальным местом разрыва предложения"""
    if not text:
        return False
    return text.endswith(('.', '!', '?', ':', ';'))


def calculate_semantic_weight(text: str) -> float:
    """Оценивает семантическую насыщенность текста"""
    if not text.strip():
        return 0.0

    try:
        doc = nlp(text)
        if not doc:
            return 0.0

        content_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        return len(content_words) / len(doc) if doc else 0.0
    except:
        # Fallback: используем простую эвристику если NLP обработка не сработала
        words = text.split()
        if not words:
            return 0.0
        # Считаем слова с большой буквы и длинные слова как содержательные
        content_words = [word for word in words if word.istitle() or len(word) > 5]
        return len(content_words) / len(words)


def find_optimal_split_point(text: str, max_length: int) -> Tuple[int, float]:
    """
    Находит оптимальную точку разрыва текста на основе семантического веса.

    Args:
        text: Текст для анализа
        max_length: Максимальная желаемая длина

    Returns:
        Tuple[int, float]: (позиция разрыва, оценка качества)
    """
    if len(text) <= max_length:
        return len(text), 1.0

    best_split = max_length
    best_score = -1.0

    # Ищем точки разрыва вокруг целевой длины
    search_start = max(0, max_length - 100)
    search_end = min(len(text), max_length + 100)

    for pos in range(search_start, search_end):
        if pos >= len(text):
            break

        # Проверяем, является ли это хорошей точкой разрыва
        if (pos < len(text) - 1 and
                is_potential_sentence_break(text[:pos]) and
                not text[pos:pos + 1].isspace() and
                not text[pos - 1:pos].isspace()):

            left_part = text[:pos]
            right_part = text[pos:]

            # Оцениваем качество разрыва
            left_weight = calculate_semantic_weight(left_part)
            right_weight = calculate_semantic_weight(right_part)

            # Комбинированная оценка (предпочтение сбалансированным разрывам)
            score = min(left_weight, right_weight) * 0.7 + (left_weight + right_weight) * 0.3

            if score > best_score:
                best_score = score
                best_split = pos

    return best_split, best_score


def split_text(
        text: str,
        max_length: int = 600,
        min_sentence_len: int = 30,
        min_chunk_len: int = 100,
        overlap_sentences: int = 1,
        language: str = "ru",
        use_semantic_splitting: bool = True  # Новая опция для семантического разбиения
) -> List[str]:
    """
    Улучшенное разбиение текста на семантически осмысленные чанки.

    Args:
        text: Текст для разбиения
        max_length: Максимальная длина чанка
        min_sentence_len: Минимальная длина предложения для отдельного рассмотрения
        min_chunk_len: Минимальная длина чанка
        overlap_sentences: Количество предложений перекрытия между чанками
        language: Язык текста ('ru' или 'en')
        use_semantic_splitting: Использовать семантическое разбиение

    Returns:
        List[str]: Список текстовых чанков
    """
    try:
        logger.debug("Начало разделения текста на чанки...")

        # Предварительная очистка текста
        text = re.sub(r'\s+', ' ', text.strip())
        if not text:
            return []

        # Если текст короткий и не требует разбиения
        if len(text) <= max_length and len(text) >= min_chunk_len:
            return [text]
        elif len(text) < min_chunk_len:
            return [text] if text else []

        # Простое разбиение для очень длинных текстов без структуры
        if not use_semantic_splitting or len(text) > 10000:
            return simple_split_text(text, max_length, min_chunk_len)

        # Улучшенное разделение на параграфы
        paragraphs = []
        current_paragraph = ""

        # Разделяем по двойным переводам строки и значимым разделителям
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = ""
                continue

            # Определяем, начинать ли новый параграф
            should_start_new = (
                    not current_paragraph or
                    line.startswith(('• ', '- ', '* ', '— ', '– ')) or  # Маркеры списков
                    line[0].isupper() and len(current_paragraph) > 100 and
                    not is_potential_sentence_break(current_paragraph) or
                    re.match(r'^\d+[\.\)]', line)  # Нумерованные списки
            )

            if should_start_new and current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = line
            else:
                if current_paragraph:
                    current_paragraph += ' ' + line
                else:
                    current_paragraph = line

        if current_paragraph:
            paragraphs.append(current_paragraph)

        chunks = []
        carryover_sentences = []  # Предложения для переноса в следующий чанк

        for paragraph in paragraphs:
            if len(paragraph) <= max_length and len(paragraph) >= min_chunk_len:
                # Если параграф уже подходящего размера
                chunks.append(paragraph)
                carryover_sentences = []  # Сбрасываем перенос
                continue
            elif len(paragraph) < min_chunk_len and chunks:
                # Очень короткий параграф - добавляем к предыдущему чанку
                chunks[-1] += ' ' + paragraph
                continue

            # Обработка длинных параграфов
            doc = nlp(paragraph)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

            if not sentences:
                continue

            # Склеивание коротких предложений с улучшенной логикой
            merged_sentences = []
            buffer = ""

            for sentence in sentences:
                # Используем семантический вес для принятия решений
                sentence_weight = calculate_semantic_weight(sentence)

                is_abbreviation = (len(sentence) <= 4 and (sentence.isupper() or sentence.istitle())) or \
                                  (len(sentence.split()) == 1 and sentence.endswith('.'))

                is_list_item = sentence.startswith(('• ', '- ', '* ', '— ', '– ')) or \
                               re.match(r'^\d+[\.\)]', sentence)

                is_short_but_important = (is_abbreviation or is_list_item or
                                          is_potential_sentence_break(sentence) or
                                          sentence_weight > 0.6 or  # Высокий семантический вес
                                          any(char.isdigit() for char in sentence))

                if len(sentence) < min_sentence_len and not is_short_but_important:
                    buffer += " " + sentence if buffer else sentence
                else:
                    if buffer:
                        # Проверяем смысловую связь с учетом семантического веса
                        buffer_weight = calculate_semantic_weight(buffer)
                        if (is_potential_sentence_break(buffer) or
                                buffer_weight > 0.5 or
                                is_abbreviation):
                            merged_sentences.append((buffer + " " + sentence).strip())
                            buffer = ""
                        else:
                            merged_sentences.append(buffer.strip())
                            buffer = sentence
                    else:
                        merged_sentences.append(sentence)

            # Обработка оставшегося буфера
            if buffer:
                if merged_sentences:
                    merged_sentences[-1] += " " + buffer.strip()
                else:
                    merged_sentences.append(buffer.strip())

            # Создание чанков с перекрытием
            current_chunk_parts = carryover_sentences.copy()
            carryover_sentences = []

            for sentence in merged_sentences:
                # Проверяем, поместится ли предложение в текущий чанк
                test_chunk = ' '.join(current_chunk_parts + [sentence])

                if len(test_chunk) <= max_length:
                    current_chunk_parts.append(sentence)
                else:
                    # Сохраняем текущий чанк
                    current_chunk_text = ' '.join(current_chunk_parts)
                    if current_chunk_text and len(current_chunk_text) >= min_chunk_len:
                        chunks.append(current_chunk_text)

                    # Определяем предложения для перекрытия
                    if overlap_sentences > 0 and len(current_chunk_parts) > overlap_sentences:
                        carryover_sentences = current_chunk_parts[-overlap_sentences:]
                    else:
                        carryover_sentences = current_chunk_parts.copy()

                    # Начинаем новый чанк с перекрытием и текущим предложением
                    current_chunk_parts = carryover_sentences.copy() + [sentence]
                    carryover_sentences = []  # Сбрасываем для следующего цикла

            # Обработка оставшихся предложений в параграфе
            if current_chunk_parts:
                current_chunk_text = ' '.join(current_chunk_parts)
                if len(current_chunk_text) >= min_chunk_len:
                    chunks.append(current_chunk_text)
                elif chunks:
                    chunks[-1] += ' ' + current_chunk_text

        # Постобработка: объединение слишком коротких чанков
        final_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if final_chunks and len(chunk) < min_chunk_len:
                # Пробуем объединить с предыдущим чанком
                combined = final_chunks[-1] + ' ' + chunk
                if len(combined) <= max_length:
                    final_chunks[-1] = combined
                else:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)

        # Фильтрация пустых чанков
        final_chunks = [chunk for chunk in final_chunks if chunk and len(chunk) >= min_chunk_len]

        # Если чанков слишком много, используем семантическое разбиение
        if not final_chunks and text:
            final_chunks = semantic_split_text(text, max_length, min_chunk_len)

        logger.debug(f"Текст разделен на {len(final_chunks)} чанков")
        return final_chunks

    except Exception as e:
        logger.error(f"Ошибка при разделении текста: {e}")
        logger.debug(traceback.format_exc())
        return simple_split_text(text, max_length, min_chunk_len)


def simple_split_text(text: str, max_length: int, min_chunk_len: int) -> List[str]:
    """Простое разбиение текста по длине"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_length
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk and len(chunk) >= min_chunk_len:
                chunks.append(chunk)
            break

        # Ищем хорошую точку разрыва
        break_pos = text.rfind('.', start, end)
        if break_pos == -1 or break_pos < start + min_chunk_len:
            break_pos = text.rfind(' ', start, end)

        if break_pos == -1 or break_pos <= start:
            break_pos = end

        chunk = text[start:break_pos + 1].strip()
        if chunk and len(chunk) >= min_chunk_len:
            chunks.append(chunk)

        start = break_pos + 1

    return chunks


def semantic_split_text(text: str, max_length: int, min_chunk_len: int) -> List[str]:
    """Семантическое разбиение текста с поиском оптимальных точек разрыва"""
    chunks = []
    start = 0

    while start < len(text):
        if len(text) - start <= max_length:
            chunk = text[start:].strip()
            if chunk and len(chunk) >= min_chunk_len:
                chunks.append(chunk)
            break

        # Находим оптимальную точку разрыва
        break_pos, score = find_optimal_split_point(text[start:start + max_length * 2], max_length)
        actual_break_pos = start + break_pos

        chunk = text[start:actual_break_pos].strip()
        if chunk and len(chunk) >= min_chunk_len:
            chunks.append(chunk)

        start = actual_break_pos

    return chunks

def indexes_creation(es, doc_id, file_name, edu_level,
                     campus, tags, url, full_text, doc_source):
    # Индексация родительского документа
    now = datetime.now(timezone(timedelta(hours=3))).isoformat(timespec="seconds")
    created_at = now
    # Если документ уже существует — сохраняем его первоначальную дату создания
    try:
        old_doc = es.get(index=PARENT_INDEX, id=doc_id)
        created_at = old_doc["_source"].get("created_at", now)
    except Exception:
        pass

    try:
        es.index(
            index=PARENT_INDEX,
            id=doc_id,
            document={
                "file_name": file_name,
                "edu_level": edu_level,
                "campus": campus,
                "topic_tag": tags,
                "url": url,
                "full_text": full_text,
                "doc_source": doc_source,
                "created_at": created_at,
                "updated_at": now
            }
        )

        logger.debug(f"Родительский документ {file_name} проиндексирован")
    except Exception as e:
        logger.error(f"Ошибка индексации родительского документа {file_name}: {e}")
        logger.debug(traceback.format_exc())
        return 0

    # Разделение на чанки и индексация
    chunks = split_text(full_text)
    for i, chunk in enumerate(chunks):
        try:
            es.index(
                index=CHUNK_INDEX,
                document={
                    "chunk_id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "text": chunk,
                    "embedding": get_embedding(chunk),
                    "topic_tag": tags,
                    "edu_level": edu_level,
                    "campus": campus
                }
            )
        except Exception as e:
            logger.error(f"Ошибка индексации чанка {i} документа {file_name}: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"Успешно проиндексирован документ: {file_name}")
    return 1


def index_documents(es: Elasticsearch):
    try:
        logger.info("Чтение метаданных документов...")
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
            file_path = os.path.join(DOCS_DIR, row['file_name'])
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
                raw_tag = raw_tag.replace("'", '"')
                try:
                    tags = json.loads(raw_tag)
                except json.JSONDecodeError:
                    tags = [tag.strip() for tag in raw_tag.split(",")]
            else:
                tags = []

            # Удаление пустых тегов
            tags = [tag for tag in tags if tag]
            file_name = row['file_name']
            doc_source = row['source']
            doc_id = hashlib.md5(row['file_name'].encode()).hexdigest()
            edu_level = [s.strip() for s in str(row['edu_level']).split(",")]
            campus = [s.strip() for s in str(row['campus']).split(",")]
            url = row['url'] if pd.notna(row['url']) else ""
            indexed_count += indexes_creation(es, doc_id, file_name,
                                             edu_level, campus, tags, url, text, doc_source)

        except Exception as e:
            logger.error(f"Ошибка обработки файла {row.get('file_name')}: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"Проиндексировано документов: {indexed_count}/{len(meta_df)}")
    return indexed_count


@app.on_event("startup")
async def startup():
    try:
        logger.info("Инициализация приложения...")

        # Подключение к Elasticsearch
        es = get_es_client(SECRET_TOKEN)

        # Создание индексов
        create_indices(es)

        # Загрузка модели
        load_model()

        # Индексация документов
        index_documents(es)

        logger.info("Инициализация завершена успешно")
    except Exception as e:
        logger.error(f"Фатальная ошибка инициализации: {e}")
        logger.debug(traceback.format_exc())
        raise RuntimeError("Application initialization failed")


def as_list(value):
    """Приводит значение к списку (в базе поле может быть строкой или списком)."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# Поиск по названию
# Порог смыслового сходства запроса и названия (от 0 до 1).
# Если находится лишнее, увеличивайте (0.85). Если не находится нужное, уменьшайте (0.78).
NAME_SEM_THRESHOLD = 0.82

# Кэш векторов названий: {doc_id: (название, вектор)}, чтобы не считать их при каждом поиске
_name_emb_cache = {}


def clean_name(name: str) -> str:
    """'Расписание_сессии.docx' -> 'Расписание сессии'"""
    name = re.sub(r"\.(docx?|txt|pdf)$", "", name or "", flags=re.I)
    return re.sub(r"[_\-]+", " ", name).strip()


def word_stems(text: str):
    """Грубая основа слова: отрезаем окончание (сессия и сессии -> сесс)."""
    stems = []
    for w in re.findall(r"\w+", text.lower()):
        if len(w) >= 6:
            stems.append(w[:-2])
        elif len(w) >= 4:
            stems.append(w[:-1])
        else:
            stems.append(w)
    return stems


def name_matches_by_words(query: str, name: str) -> bool:
    """Все слова запроса (по основам) должны найтись в начале каких-то слов названия."""
    name_words = re.findall(r"\w+", name.lower())
    q_stems = word_stems(query)
    if not q_stems:
        return False
    return all(any(nw.startswith(qs) for nw in name_words) for qs in q_stems)


def prefixed_embedding(text: str, prefix: str):
    """e5 лучше работает, если к тексту добавить 'query: ' или 'passage: '."""
    if model is None:
        load_model()
    emb = model.encode(prefix + text, convert_to_numpy=True)
    return normalize(emb.reshape(1, -1))[0]


def name_embedding(doc_id: str, name: str):
    cached = _name_emb_cache.get(doc_id)
    if cached and cached[0] == name:      # название не менялось, берём из кэша
        return cached[1]
    vec = prefixed_embedding(name, "passage: ")
    _name_emb_cache[doc_id] = (name, vec)
    return vec


def find_docs_by_name(es, req):
    """Ищет документы по названию: сначала по словам, затем по смыслу."""
    q = req.query.strip()
    if len(q) < 3:
        return []

    res = es.search(index=PARENT_INDEX, body={"size": 1000, "query": {"match_all": {}}})
    query_vec = None
    found = []

    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        raw_name = src.get("file_name") or ""
        name = clean_name(raw_name)

        # Те же фильтры, что выбрал пользователь
        if req.edu_level and req.edu_level not in as_list(src.get("edu_level")):
            continue
        if req.campus and req.campus not in as_list(src.get("campus")):
            continue
        if req.topic_tag and not set(req.topic_tag) & set(as_list(src.get("topic_tag"))):
            continue

        # 1) совпадение по словам (с учётом окончаний)
        if q.lower() in raw_name.lower() or name_matches_by_words(q, name):
            score = 1.0
        else:
            # 2) совпадение по смыслу
            if query_vec is None:
                query_vec = prefixed_embedding(q, "query: ")
            score = float(np.dot(query_vec, name_embedding(hit["_id"], name)))
            if score < NAME_SEM_THRESHOLD:
                continue

        full_text = src.get("full_text", "")
        found.append({
            "doc_id": hit["_id"],
            "score": score,
            "file_name": src.get("file_name"),
            "url": src.get("url", ""),
            "tags": src.get("topic_tag", []),
            "chunks": [{"text": full_text[:200], "chunk_score": score, "rerank_score": score}],
            "full_text": full_text,
            "doc_source": src.get("doc_source", ""),
            "created_at": src.get("created_at"),
            "updated_at": src.get("updated_at"),
            "rerank_score": score,
        })

    found.sort(key=lambda d: d["rerank_score"], reverse=True)
    return found



@app.post("/search")
def search(req: SearchRequest, token: str = Header(...)):
    try:
        es = get_es_client(token)
        query_emb = get_embedding(req.query)

        # Построение фильтров
        filters = []
        if req.edu_level:
            filters.append({"term": {"edu_level": req.edu_level}})
        if req.campus:
            filters.append({"term": {"campus": req.campus}})
        if req.topic_tag:
            filters.append({"terms": {"topic_tag": req.topic_tag}})

        # Улучшенный запрос с гибридным поиском
        query = {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": [{"match_all": {}}],
                        "should": [
                            {
                                "multi_match": {
                                    "query": req.query,
                                    "fields": ["text^1.5",
                                               "title^1.5"
                                               ],
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "minimum_should_match": 0
                    }
                },
                "script": {
                    "source": """
                        double similarity = cosineSimilarity(params.query_vector, 'embedding');
                        double score = (similarity + 1.0) / 2.0;

                        if (params._source.text != null && 
                            params._source.text.toLowerCase().contains(params.query_text.toLowerCase())) {
                            score *= 1.2;
                        }

                        return score;
                    """,
                    "params": {
                        "query_vector": query_emb,
                        "query_text": req.query
                    }
                },
                "min_score": 0.0
            }
        }

        response = es.search(index=CHUNK_INDEX, body={
            #"size": req.count_doc_return * 10,  # Достаточно для группировки
            "size": max(req.count_doc_return, req.count_doc_rerank) * 20,  # берём достаточно фрагментов, чтобы покрыть все документы для reranker'а
            "query": query,
            "highlight": {
                "fields": {
                    "text": {
                        "fragment_size": req.fragment_size,
                        "number_of_fragments": req.number_fragments_in_chunk,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            }
        }, timeout="100s")

        # Группировка чанков по документам с сохранением информации о score
        chunks_by_doc = {}
        seen_chunk_ids = set()  # Защита от дубликатов

        for hit in response["hits"]["hits"]:
            doc_id = hit["_source"]["doc_id"]
            chunk_id = f"{doc_id}_{hit['_source'].get('chunk_index', 0)}"

            # Пропускаем дубликаты
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            highlight = hit.get("highlight", {}).get("text", [hit["_source"]["text"]])[0]

            chunk_info = {
                "text": highlight,
                "chunk_score": round(hit["_score"], 4)
            }

            chunks_by_doc.setdefault(doc_id, []).append(chunk_info)

        if not chunks_by_doc:
            return []
        
        # Получение метаданных документов
        parent_docs = es.mget(index=PARENT_INDEX, body={"ids": list(chunks_by_doc.keys())})
        doc_meta = {doc["_id"]: doc["_source"] for doc in parent_docs["docs"] if doc["found"]}

        # Формирование результатов (БЕЗ ограничения чанков)
        results = []
        for doc_id, chunks in chunks_by_doc.items():
            if doc_id not in doc_meta:
                continue

            meta = doc_meta[doc_id]
            doc_score = max(chunk["chunk_score"] for chunk in chunks)

            results.append({
                "doc_id": doc_id,
                "score": round(doc_score, 4),
                "file_name": meta.get("file_name"),
                "url": meta.get("url", ""),
                "tags": meta.get("topic_tag", []),
                "chunks": chunks,
                "full_text": meta.get("full_text", ""),
                "doc_source": meta.get("doc_source", ""),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at")
            })

        # Обрезаем кандидатов по score из ES
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:req.count_doc_rerank]

        # -----------------------------
        # Реранкер по чанкам
        # -----------------------------
        pairs = []
        pair_to_loc = []

        for doc_idx, doc in enumerate(results):
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                chunk_text = chunk.get("text", "")
                pairs.append((req.query, chunk_text))
                pair_to_loc.append((doc_idx, chunk_idx))

        if pairs:
            scores_reranker = reranker.predict(pairs)

            # Проставляем rerank_score в чанки
            doc_best_scores = {i: float("-inf") for i in range(len(results))}
            for i, score in enumerate(scores_reranker):
                doc_idx, chunk_idx = pair_to_loc[i]
                score = float(score)
                results[doc_idx]["chunks"][chunk_idx]["rerank_score"] = round(score, 6)
                if score > doc_best_scores[doc_idx]:
                    doc_best_scores[doc_idx] = score

            # Добавляем rerank_score документам (max по чанкам)
            for doc_idx, best_score in doc_best_scores.items():
                if best_score == float("-inf"):
                    best_score = 0.0
                results[doc_idx]["rerank_score"] = round(best_score, 6)
        else:
            for doc in results:
                doc["rerank_score"] = 0.0

        # Сортировка по rerank_score
        results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)


        # Минимальная оценка reranker'а, при которой документ считается подходящим.
        # Подобрана по тестовым запросам. Если полезные документы пропадают - уменьшайте.
        # Если появляется мусор - увеличивайте.
        MIN_RERANK_SCORE = 0.005

        results = [
            doc for doc in results
            if doc.get("rerank_score", 0.0) >= MIN_RERANK_SCORE
        ]

        # Документы, найденные по названию, показываем первыми
        by_name = find_docs_by_name(es, req)
        name_ids = {doc["doc_id"] for doc in by_name}
        results = by_name + [doc for doc in results if doc["doc_id"] not in name_ids]

        return results[:req.count_doc_return]

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_doc")
def add(req: AddRequest, token: str = Header(...)):
    try:
        es = get_es_client(token)
        flag = indexes_creation(es, req.doc_id, req.file_name, req.edu_level,
                     req.campus, req.topic_tag, req.url, req.full_text, req.doc_source)
        if flag:
            logger.info(f"Успешно добавлен документ: {req.file_name}")
            return {"file_name": req.file_name, "added": True}
        else:
            logger.info(f"Не удалось добавить документ: {req.file_name}")
            return {"file_name": req.file_name, "added": False}

    except Exception as e:
        logger.error(f"Ошибка добавления документа {req.file_name}: {e}")
        logger.debug(traceback.format_exc())
        return {"file_name": req.file_name, "added": False}

@app.post("/delete_doc")
def delete(req: DeleteRequest, token: str = Header(...)):
    try:
        es = get_es_client(token)
        doc_is_deleted = False
        # Получить doc_id
        doc_id = hashlib.md5(req.file_name.encode()).hexdigest()
        try:
            es.delete(index=PARENT_INDEX, id=doc_id)
            logger.info(f"Документ с id '{doc_id}' удалён из {PARENT_INDEX}")
            doc_is_deleted = True
        except Exception as e:
            logger.info(f"Документ с id '{doc_id}' не найден в {PARENT_INDEX} — не удалён")
        # --- Удалить чанки с этим doc_id ---
        # Поиск всех чанков
        query = {
            "query": {
                "term": {
                    "doc_id": doc_id
                }
            }
        }

        # Получить чанки (только _id)
        res = es.search(index=CHUNK_INDEX, body=query, scroll="1m", size=1000, timeout="100s")
        scroll_id = res["_scroll_id"]
        hits = res["hits"]["hits"]

        # Список для удаления
        to_delete = []

        while hits:
            for hit in hits:
                to_delete.append({
                    "_op_type": "delete",
                    "_index": CHUNK_INDEX,
                    "_id": hit["_id"]
                })

            res = es.scroll(scroll_id=scroll_id, scroll="1m")
            hits = res["hits"]["hits"]

        if to_delete:
            bulk(es, to_delete)
            logger.info(f"Удалено {len(to_delete)} чанков из {CHUNK_INDEX}")
        else:
            logger.info("Чанки не найдены.")

        return {"file_name": req.file_name, "doc_is_deleted": doc_is_deleted, "num_deleted_chunks": len(to_delete)}
    except Exception as e:
        logger.error(f"Ошибка удаления документа {req.file_name}: {e}")
        logger.debug(traceback.format_exc())
        return {"file_name": req.file_name, "doc_is_deleted": False, "num_deleted_chunks": 0}

@app.post("/get_logs")
def get_logs(req: GetLogs):
    try:
        PATH = ACTUAL_LOGS_PATH + req.log_name
        # Загрузка json файла
        with open(PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            "logs": data,
            "time": data[-1].get("Дата уточняющего вопроса", data[-1].get("Дата вопроса"))
        }
    except Exception as e:
        logger.error(f"Ошибка получения логов : {e}")
        logger.debug(traceback.format_exc())
        return {"error": f"Ошибка получения логов: {e}"}


@app.get("/documents")
def list_documents(
        token: str = Header(...),
        size: int = 1000,
        edu_level: Optional[str] = None,
        campus: Optional[str] = None,
        topic_tag: Optional[List[str]] = Query(None),
):
    """Список документов (без полного текста). Можно фильтровать по уровню образования, кампусу и тегам."""
    try:
        es = get_es_client(token)

        filters = []
        if edu_level:
            filters.append({"term": {"edu_level": edu_level}})
        if campus:
            filters.append({"term": {"campus": campus}})
        if topic_tag:
            filters.append({"terms": {"topic_tag": topic_tag}})

        res = es.search(
            index=PARENT_INDEX,
            body={
                "query": {"bool": {"must": [{"match_all": {}}], "filter": filters}},
                "_source": {"excludes": ["full_text"]},
                "size": size
            }, timeout="100s"
        )
        documents = []
        for hit in res["hits"]["hits"]:
            item = hit["_source"]
            item["doc_id"] = hit["_id"]
            documents.append(item)
        return {"total": len(documents), "documents": documents}
    except Exception as e:
        logger.error(f"Ошибка получения списка документов: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка получения списка документов")


@app.get("/tags")
def list_tags(token: str = Header(...)):
    """Все теги, которые сейчас есть в базе (для выпадающего списка фильтра)"""
    try:
        es = get_es_client(token)
        res = es.search(
            index=PARENT_INDEX,
            body={"size": 0, "aggs": {"tags": {"terms": {"field": "topic_tag", "size": 500}}}}, timeout="100s"
        )
        tags = sorted(b["key"] for b in res["aggregations"]["tags"]["buckets"])
        return {"tags": tags}
    except Exception as e:
        logger.error(f"Ошибка получения тегов: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка получения тегов")


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, token: str = Header(...)):
    """Получить один документ целиком по его id"""
    es = get_es_client(token)
    try:
        res = es.get(index=PARENT_INDEX, id=doc_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Документ не найден")
    doc = res["_source"]
    doc["doc_id"] = res["_id"]
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, token: str = Header(...)):
    """Удалить документ напрямую по id (использует фронтенд)"""
    es = get_es_client(token)
    doc_is_deleted = False
    try:
        es.delete(index=PARENT_INDEX, id=doc_id)
        doc_is_deleted = True
    except Exception:
        pass

    query = {"query": {"term": {"doc_id": doc_id}}}
    res = es.search(index=CHUNK_INDEX, body=query, scroll="1m", size=1000, timeout="100s")
    scroll_id = res["_scroll_id"]
    hits = res["hits"]["hits"]
    to_delete = []
    while hits:
        for hit in hits:
            to_delete.append({"_op_type": "delete", "_index": CHUNK_INDEX, "_id": hit["_id"]})
        res = es.scroll(scroll_id=scroll_id, scroll="1m")
        hits = res["hits"]["hits"]
    if to_delete:
        bulk(es, to_delete)

    return {"doc_id": doc_id, "doc_is_deleted": doc_is_deleted, "num_deleted_chunks": len(to_delete)}


class EditRequest(BaseModel):
    file_name: str
    edu_level: list[str]
    campus: list[str]
    topic_tag: list[str]
    url: str
    full_text: str
    doc_source: str = ""


@app.put("/documents/{doc_id}")
def edit_document(doc_id: str, req: EditRequest, token: str = Header(...)):
    """Редактировать документ: обновить метаданные и полностью пересоздать чанки"""
    es = get_es_client(token)

    try:
        es.get(index=PARENT_INDEX, id=doc_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Документ не найден")

    # Удаляем старые чанки — текст мог измениться, старые эмбеддинги не актуальны
    query = {"query": {"term": {"doc_id": doc_id}}}
    res = es.search(index=CHUNK_INDEX, body=query, scroll="1m", size=1000, timeout="100s")
    scroll_id = res["_scroll_id"]
    hits = res["hits"]["hits"]
    to_delete = []
    while hits:
        for hit in hits:
            to_delete.append({"_op_type": "delete", "_index": CHUNK_INDEX, "_id": hit["_id"]})
        res = es.scroll(scroll_id=scroll_id, scroll="1m")
        hits = res["hits"]["hits"]
    if to_delete:
        bulk(es, to_delete)

    # Пересоздаём документ и чанки (created_at сохранится автоматически, updated_at обновится)
    indexes_creation(es, doc_id, req.file_name, req.edu_level, req.campus,
                      req.topic_tag, req.url, req.full_text, req.doc_source)

    return {"doc_id": doc_id, "updated": True}




@app.get("/")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)