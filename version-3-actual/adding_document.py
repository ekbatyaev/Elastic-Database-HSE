import hashlib
import json
import logging
import os
import traceback
import requests
import fitz
import pandas as pd
from docx import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

META_PATH = "add_meta.xlsx"
DOCS_DIR = "add_folder"

def request_add(doc_id, file_name, edu_level, doc_source, campus, topic_tag, url_add, full_text):

    # Указываем URL FastAPI сервера

    url = "url"


    # Пример данных для запроса
    add_request = {
        "doc_id": doc_id,
        "file_name": file_name,
        "edu_level": edu_level,
        "campus": campus,
        "topic_tag": topic_tag,
        "url": url_add,
        "doc_source": doc_source,
        "full_text": full_text
    }

    # Указываем секретный токен для авторизации
    headers = {
        "token": "token"
    }
    print(add_request)
    # Выполняем POST-запрос
    response = requests.post(url, json=add_request, headers=headers)

    # Проверяем статус ответа
    if response.status_code == 200:
        # Выводим результаты поиска
        results = response.json()
        return results


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


try:
    logger.info("Чтение метаданных документов...")
    meta_df = pd.read_excel(META_PATH)
    print(meta_df)
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

        doc_id = hashlib.md5(row['file_name'].encode()).hexdigest()
        edu_level = [s.strip() for s in str(row['edu_level']).split(",")]
        campus = [s.strip() for s in str(row['campus']).split(",")]
        url = row['url'] if pd.notna(row['url']) else ""
        doc_source = row['source']
        print(request_add(doc_id, row['file_name'], edu_level, doc_source, campus, tags, url, text))

    except Exception as e:
        logger.error(f"Ошибка чтения метаданных: {e}")
        logger.debug(traceback.format_exc())
        raise