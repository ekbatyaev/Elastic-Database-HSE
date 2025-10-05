import logging
import os
import traceback
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

META_PATH = "delete_meta.xlsx"
DOCS_DIR = "del_folder"


def request_delete(file_name):

    # Указываем URL FastAPI сервера

    url = "url"

    # Пример данных для запроса
    search_request = {
        "file_name": file_name
    }

    # Указываем секретный токен для авторизации
    headers = {
        "token": "token"
    }

    # Выполняем POST-запрос
    response = requests.post(url, json=search_request, headers=headers)

    # Проверяем статус ответа
    if response.status_code == 200:
        # Выводим результаты поиска
        results = response.json()
        return results


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
        print(request_delete(row['file_name']))

    except Exception as e:
        logger.error(f"Ошибка чтения метаданных: {e}")
        logger.debug(traceback.format_exc())
        raise