import requests


url = "URL"

question = ("что такое лмс?")

# Пример данных для запроса
search_request = {
    "query": question,
    "edu_level": "бакалавриат",
    "campus": "Москва",
    "topic_tag": ["Учебный процесс"],
    "bm25_weight": 0.3,
    "embed_weight": 0.7,
    "fragment_size": 800,
    "count_doc_return": 3
}

# Указываем секретный токен для авторизации
headers = {
    "token": "SECRET_TOKEN"
}

# Выполняем POST-запрос
response = requests.post(url, json=search_request, headers=headers)


if response.status_code == 200:

    results = response.json()
    print(results)
