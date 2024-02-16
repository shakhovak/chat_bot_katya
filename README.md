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

### Данные для обучения bi-encoder 
На основе обработанных данных готовятся данные для обучения bi-encoder. Так как обучать я решила обучать модель c использованием triplet loss, то данные преобразуются в триплеты:
- ancor - комбинация контекста и вопроса
- positive - ответ на вопрос из сценария
- negative - случайно подобранный ответ. Данные для негативных семплов были скачаны также с Kaggle и можно посмотреть их по [ссылке - rickmorty](https://www.kaggle.com/datasets/andradaolteanu/rickmorty-scripts) + [star_wars](https://www.kaggle.com/datasets/xvivancos/star-wars-movie-scripts?rvi=1).

Функции для обработки данных находятся в файле ```utils.py```:
1. ```data_prep_biencoder``` - считывает файлы, рандомно объединяет их с правильными ответами в отдельный стобец и затем сохраняет в pkl формат.

### Данные для обучения reranker
Эти же данные будут использованы для обучения модели-reranker для переранжирования вариантов ответа. Исходные данные будут приняты как правильные с лейблом 0 и будут дополнены репликами из других сериалов в качестве ответа и помечены лейблом 1. Данные для негативных семплов были скачаны также с Kaggle и можно посмотреть их по [ссылке - rickmorty](https://www.kaggle.com/datasets/andradaolteanu/rickmorty-scripts) + [star_wars](https://www.kaggle.com/datasets/xvivancos/star-wars-movie-scripts?rvi=1).

Данные для reranker включают контекст+вопрос+ответ, разделенные специальным токеном [SEP]. Подготовлено примерно 16 тыс. семплом для обучения, разбивка по классам 50:50.

Функции для обработки данных находятся в файле ```utils.py```:
1. ```read_files_negative``` - считывает файлы для негативных семплов, объединяет с положительными и сохраняет в pkl формат.


## Архитектура чат-бота

Схематично процесс работы чат-бота представлен на рисунке ниже.

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/90f460a2-1062-4b55-b2eb-83961e840709)


**База данных реплик** включает векторизованные при помощи модели [обученного_энкодера](https://huggingface.co/Shakhovak/chatbot_sentence-transformer) скрипты, включающие контекст и вопрос. 

Отбор реплик из базы данных будет проводиться в 2 этапа:
- отбор похожих по косинусной близости контекста+вопрос из созданной векторной базы данных. В результате отбирается 20 кандидатов с максимальным скорингом похожести на пользовательский контекст+вопрос.
- классификация моделью-reranker на основе Bert полученных кандидатов на предмет того, является ли подобранный ответ продолжением вопроса с контекстом или нет. Среди кандидатов отбираются те, которые были классифицированы как относящиеся к классу 0 (ответ является продолжением) и ранжируются по скорингу-уверенности модели в отнесении образца к нужному классу. Если же все реплики были классифицированы как относящиеся к 1 классу (ответ НЕ является продолжением вопроса и контекста), то в ответ подается топ-1 из отобранных по косинусной близости.

:bulb: **Intent классфикатор** взят готовый из библиотеки [DialogTag](https://pypi.org/project/DialogTag/) и использовался как для разметки исходных данных, так и получаемого от пользователя высказывания. на основе полученного намерения фильтруется пул отобранных кандидатов. Также тег с намерением добавляется в эмбединги, которые использует bi-encoder.

В основе модели  :bulb: **bi-encoder** ```distilroberta-base ```, обученная на описанных раннее данных в виде триплетов. Для обучения я воспользовалась библиотекой sentence transformers. Обучение основано на Triplet Loss Function, которая минимизиурет расстояние между якорным предложением и правильным ответом и максимизирует между якорным и неправильным ответом (см. рис ниже). 

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/6e5129c2-bab7-49d7-aac1-20bb53259c2f)

Оценивается модель по точности (accuracy) выявления случаев, когда близость между якорным текстом и правильным ответом больше, чем между якорным и нейтральным. Необученная distilroberta-base давала 65%, после обучения метрика стала на уровне 95%

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/94f01abf-153e-4c62-96c4-ebac81c090ff)

Посмотреть ноутбук с обученем модел можно [здесь](https://github.com/shakhovak/chat_bot_katya/blob/master/train_models/bi_encoder_train2.ipynb) . Модель загружена в мой репозиторий на Hugging Face ([ссылка](https://huggingface.co/Shakhovak/chatbot_sentence-transformer)) и уже оттуда будет использоваться в инференсе.

В основе модели :bulb: **re-ranker** ```bert-base-uncased```, обученная на подготовленных ранее данных. Классификация оценивалась при помощи accuracy, было сделано 3 подхода к обучению. Графики обучения можно посмотреть в wandb по [ссылке](https://wandb.ai/shakhova/reranker_train?workspace=user-katya_shakhova) . Ниже принт-скрин графиков обучения.

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
|
├───train_models - ноутбуки с обучением моделей
├───templates - оформление веб-интерфейса
│       chat.html
├───static - оформление веб-интерфейса
│       style.css
├───data
│       low_score_scripts.pkl - сценарии при низких скорах похожести
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
В целом чат-бот должен оцениваться по релевантности реплик в контексте диалога, поэтому здесь основной все-таки будет пользовательская оценка. 

Я попробовала посмотреть, как будет отвечать чат-бот при применении разных видов энкодеров:
- [sentence-transformers/all-mpnet-base-v2]() - готовый обученный энкодер
- [sentence-transformers/LaBSE]()- готовый обученный энкодер
- [Shakhovak/chatbot_sentence-transformer]() - энкодер, который обучила на данных, описанных выше для bi-encoder

Я выбрала одинаковые реплики и посмотрела, как работает retrieval (см. таблицу ниже)

| Encoder | Hi man!  | Any plans for today?  | What are you talking about?  |Let’s talk about Leonard  |
| :---:   | :---: | :---: |:---: |:---: |
| sentence-transformers/all-mpnet-base-v2| Hello,my friend.  |What does it mean?|You askedmy friend if she wanted to hear something weird.|Again, urban slang. In which, I believe I’ m gaining remarkable fluency. So, could you repeat?|
| sentence-transformers/LaBSE | Hello,my friend. | Yes?  |My plan was to jump out at the state line, but one of my nose plugs fell into the toilet.  |Nothing. I say nothing.  |
| Shakhovak/chatbot_sentence-transformer | Hello. So I guess you’ re really holding up the other four fingers?   | It’ s called fitting in. By the way, good luck.   |You clearly weren’ t listening to my topic sentence, get your women in line! You make them apologize tomy friend and set things right. I am a man of science, not someone’ s snuggle bunny!  |Then it hits her. How is she going to survive? I mean, she has no prospects, no marketable skills. And then one day, she meets a group of geniuses and their friend Howard.  |

Интересно, что при использовании эмбедингов от модели, обученных на данных по Шелдону, скоры похожести стали выше, чем при использовании более общих эмбедингов. 

Среди выше приведенных примеров сложно сказать, какой лучше. Я решила оставить обученную на даннаом датасете модель и добавить ограничение на intent и минимальный уровень похожести ответа перед передачей данных в re-ranker, чтобы добавить немного детерминированности в диалог.

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






