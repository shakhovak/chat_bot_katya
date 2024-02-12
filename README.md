# HW1 Retrieval-based чат-бот

**Задание**: Необходимо разработать чат-бота, используя подход retrieval-based. Бот должен вести диалог как определенный персонаж сериала, имитируя стиль и манеру конкретного персонажа сериала. Важно учесть особенности речи и темы, которые поднимает персонаж, его типичные реакции.

## Данные
В качестве основы для чат-бота я взяла скрипты к сериалу "Теория большого взрыва", которые есть на Kaggle и можно загрузить по [ссылке](https://www.kaggle.com/code/lydia70/big-bang-theory-tv-show/input).

Основной персонаж - Шелдон Купер :)

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/de2bed9e-e2a6-46a0-a24a-6cb94c4f0f61)


Данные на kaggle уже удобно разделены на реплики каждого персонажа и на отдельные сцены (ниже принт-скрин данных)

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/100d2802-4837-40d9-95ad-c41034e184fb)

Обработка данных подразумевает следующие шаги:
- отбор реплик персонажа в качестве ответов / answers. Именно из этих реплик будет выбирать бот свой ответ на высказывание пользователя.
- выделение предшествующей фразы как вопроса. Если это фраза первая в сцене, то это поле будет пустым.
- отбор предыдущих реплик как контекста диалога (ограничение не более 5 фраз в контексте). Если фраза первая в сцене, то контекст также будет пустым. Контекст - идущие подряд предложения. Я не стала разбивать на диалоги, как предлагалось на семинаре.
- сохранение файла в pickle формат для последующего использования алгоритмом

После обработки получилось более 11 тыс. образцов, которые могут быть использованы ботов в диалоге.

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/1203eff9-cd4a-41e0-8016-1f1fbb700032)

Функции для обработки данных находятся в файле ```utils.py```:
1. ```scripts_rework``` - перерабатывает файл, как описано выше и сохраняет их в pkl формате
2. ```encode_df_save``` - использует переработанный файл и векторизует его, а вектора уже сохраняет в базу

### Данные для обучения reranker
Эти же данные будут использованы для обучения модели-reranker для переранжирования вариантов ответа. Исходные данные будут приняты как правильные с лейблом 0 и будут дополнены репликами из других сериалов в качестве ответа и помечены лейблом 1. Данные для негативных семплов были скачаны также с Kaggle и можно посмотреть их по [ссылке - rickmorty](https://www.kaggle.com/datasets/andradaolteanu/rickmorty-scripts) + [star_wars](https://www.kaggle.com/datasets/xvivancos/star-wars-movie-scripts?rvi=1).

Данные для reranker включают контекст+вопрос+ответ, разделенные специальным токеном [SEP]. Подготовлено примерно 16 тыс. семплом для обучения, разбивка по классам 50:50.

Функции для обработки данных находятся в файле ```utils.py```:
1. ```read_files_nefgative``` - считывает файлы для негативных семплов, объединяет с положительными и сохраняет в pkl формат.


## Архитектура чат-бота

Схематично процесс работы чат-бота представлен на рисунке ниже.

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/80080c94-b561-4537-b414-fa4e28abb3a4)

**База данных реплик** включает векторизованные при помощи модели [Labse](https://huggingface.co/sentence-transformers/LaBSE?library=sentence-transformers) скрипты, включающие контекст и вопрос. 

Отбор реплик из базы данных будет проводиться в 2 этапа:
- отбор похожих по косинусной близости контекста+вопрос из созданной векторной базы данных. В результате отбирается 20 кандидатов с максимальным скорингом похожести на пользовательский контекст+вопрос.
- классификация моделью-reranker на основе Bert полученных кандидатов на предмет того, является ли подобранный ответ продолжением вопроса с контекстом или нет. Среди кандидатов отбираются те, которые были классифицированы как относящиеся к классу 0 (ответ является продолжением) и ранжируются по скорингу-уверенности модели в отнесении образца к нужному классу. Если же все реплики были классифицированы как относящиеся к 1 классу (ответ НЕ является продолжением вопроса и контекста), то в ответ подается топ-1 из отобранных по косинусной близости.

В основе модели re-ranker ```bert-base-uncased```, обученная на подготовленных ранее данных. Классификация оценивалась при помощи accuracy, было сделано 3 подхода к обучению. Графики обучения можно посмотреть в wandb по [ссылке](https://wandb.ai/shakhova/reranker_train?workspace=user-katya_shakhova) . Ниже принт-скрин графиков обучения.

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/2ae7c305-0e23-45e7-baa8-e8390fc55b48)

Высоких показателей не удалось достичь, финальная точность модели варьируется от 78 до 80%. Также видно, что на 3-ей эпохе модель уже переобучилась.

Модель выгружена на Hugging Face ([ссылка](https://huggingface.co/Shakhovak/RerankerModel_chat_bot)) и уже оттуда будет использоваться в инференсе.

## Структура репозитория

```bash
│   README.md - отчет для ДЗ
│   requirements.txt
│   Dockerfile
|
│   retrieve_bot.py - основной файл алгоритма
│   utils.py - вспомогательные функции в т.ч. для предобработки данных
|   app.py - для запуска UI c flask
|   train_model.ipynb - ноутбук с обученим модели 
|
├───templates - оформление веб-интерфейса
│       chat.html
├───static - оформление веб-интерфейса
│       style.css
├───data
│       scripts_for_reranker.pkl - обучающие данные для reranker
│       scripts_vectors.pkl - база данных контекст+воппрос на основе векторов LaBSe
│       scripts.pkl - исходные данные
```

## Реализация web-сервиса

Реализован чат на основе Flask, входной скрипт ```app.py```, который выстраивает графический интерфейс - за основу взят дизайн с [tutorial](https://www.youtube.com/watch?v=70H_7C0kMbI&list=WL&index=4&t=105s), создает инстант класса ChatBot, загружает файлы и модели. Также есть Dockerfile, который я использовала, чтобы развернуть сервис на сервере hugging face. 

> [!IMPORTANT]
> - Попробовать поговорить с Шелдоном можно по [ссылке](https://huggingface.co/spaces/Shakhovak/Sheldon_Retrieval_chat_bot). Это бесплатный ресурс, иногда работает медленно :(
> - Дополнительно развернула docker image с ботом наа ВМ d Yandex Cloud по [ссылке](http://84.201.157.133:5000). Здесь платный ресурс, но тоже без GPU. Субъективно, на платной машине быстрее работает. 


Хочу обратить внимание, что бесплатное размещение не включает GPU, только CPU, поэтому инференс работает медленнее, чем на локальном компьютере. Для ускорения я сократила число кандидатов до 5. Кроме того, сервер на Hugging face работает в течение 48 часов после последнего посещения, поэтому при проверке может понадобиться еще раз запустить сервер.

Один из минусов моей реализации - отсутствие сессий и обновления контекста для каждого пользователя/сессии. Я оставила только ограничения по общему размеру контекста (не более 5 предложений).

## Оценка качества чат-бота
В целом чат-бот должен оцениваться по релевантности реплик в контексте диалога, поэтому здесь основной все-таки будет пользовательская оценка. В процессе работы я смотрела, как отвечает бот только на основе косинусной близости, а потом как меняются фразы при добавлении модели reranker. Субъективно, сильных улучшений я не заметила. В целом оцениваю результат как средний, нужна все-таки генеративная модель, так как контекста не так много. 

Пример диалога:

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/a0181682-65f8-443c-9bb6-54f2552825a5)


А вот, что было изначально по косинусной близости:

| Пользователь | Reranker   | Начальный ответ    |
| :---:   | :---: | :---: |
| Hi, Sheldon! Any plans for today?| I heard a noise.  | Hello. I’m here for my haircut with Mr. D’Onofrio.|
| What are you talking about? | Oh, I don’t know, uh, weather, uh, fish you could do carpentry with, why Leonard is such an attractive and desirable boyfriend. Yeah, pick one, your choice.   | Earlier, I came here to surprise you. I looked in the window and I saw you with a man.   |
| Let's talk about Leonard  | You will?   | You will?   |



## Начать работу с чатом

Для установки чата, можно воспользоваться docker image, который я сохранила на публичный репозиторий в dockerhub. C помощью команды ```docker pull shakhovak/chat2:latest``` скачать образ и запустить ```sudo docker run -it --name chat -p 5000:5000 --rm shakhovak/chat2``` .

> [!WARNING]
> Обращаю внимание, что в image нет начальных данных, а только обработанные файлы pickle. Чтобы запустить обработку начальных данных, нужно скачать файлы в папку data (все файлы star wars в подпапку star_wars) и запустить функции по их предобработке.

<hr>

### Шпаргалка как развернуть docker image в Yandex Cloud
1. Создать ВМ и убедиться, что у нее открыт наружу нужный порт (в случае с ботом - 5000). Машину создала на Debian
2. Установить на ВМ docker enginе. Инструкция вот [здесь](https://docs.docker.com/engine/install/debian/) для debian. Основные команды:
  - удалить потенциальные конфликты  
  - setup docker apt repository
  -  установить докер
3. Залогиниться на docker hub ``` sudo docker login ``` и запулить докер образ на ВМ ```sudo docker pull shakhovak/chat2:latest```
4. Запустить образ на ВМ  ```sudo docker run -it --name chat -p 5000:5000 --rm shakhovak/chat2```

<hr>






