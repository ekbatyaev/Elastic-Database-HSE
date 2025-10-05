import requests

# Указываем URL FastAPI сервера

url = "url"

question = ("вопрос")

# Пример данных для запроса
search_request = {
    "query": question,
    "edu_level": "бакалавриат",
    "campus": "Нижний Новгород",
    "topic_tag": ["Учебный процесс"],
    "bm25_weight": 0.3,
    "embed_weight": 0.7,
    "fragment_size": 800,
    "count_doc_return": 3
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
    print(results)