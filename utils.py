import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pandas as pd
import pickle
import random
from nltk.tokenize import word_tokenize
import string


def encode(texts, model, intent, contexts=None, do_norm=True):
    """function to encode texts for cosine similarity search"""

    question_vectors = model.encode(texts)
    context_vectors = model.encode("".join(contexts))
    intent_vectors = model.encode(intent)

    return np.concatenate(
        [
            np.asarray(context_vectors),
            np.asarray(question_vectors),
            np.asarray(intent_vectors),
        ],
        axis=-1,
    )


# ===================================================


def cosine_sim(data_vectors, query_vectors) -> list:
    """returns list of tuples with similarity score and
    script index in initial dataframe"""

    data_emb = sparse.csr_matrix(data_vectors)
    query_emb = sparse.csr_matrix(query_vectors)
    similarity = cosine_similarity(query_emb, data_emb).flatten()
    ind = np.argwhere(similarity)
    match = sorted(zip(similarity, ind.tolist()), reverse=True)

    return match


# ===================================================


def scripts_rework(path, character, tag_model):
    """this functions split scripts for queation, answer, context,
    picks up the cahracter and saves data in pickle format"""

    df = pd.read_csv(path)

    # split data for scenes
    count = 0
    df["scene_count"] = ""
    for index, row in df.iterrows():
        if index == 0:
            df.iloc[index]["scene_count"] = count
        elif row["person_scene"] == "Scene":
            count += 1
            df.iloc[index]["scene_count"] = count
        else:
            df.iloc[index]["scene_count"] = count

    df = df.dropna().reset_index()

    # rework scripts to filer by caracter utterances and related context
    scripts = pd.DataFrame()
    for index, row in df.iterrows():
        if (row["person_scene"] == character) & (
            df.iloc[index - 1]["person_scene"] != "Scene"
        ):
            context = []
            for i in reversed(range(2, 5)):
                if (df.iloc[index - i]["person_scene"] != "Scene") & (index - i >= 0):
                    context.append(df.iloc[index - i]["dialogue"])
                else:
                    break
            new_row = {
                "answer": row["dialogue"],
                "question": df.iloc[index - 1]["dialogue"],
                "context": context,
            }

            scripts = pd.concat([scripts, pd.DataFrame([new_row])])

        elif (row["person_scene"] == character) & (
            df.iloc[index - 1]["person_scene"] == "Scene"
        ):
            context = []
            new_row = {"answer": row["dialogue"], "question": "", "context": context}
            scripts = pd.concat([scripts, pd.DataFrame([new_row])])
    # load reworked data to pkl
    scripts = scripts[scripts["question"] != ""]
    scripts["answer"] = scripts["answer"].apply(lambda x: change_names(x))
    scripts["tag"] = scripts[["answer", "question"]].apply(
        lambda test_scripts: intent_classification(
            test_scripts["question"], test_scripts["answer"], tag_model
        ),
        axis=1,
    )
    scripts = scripts.reset_index(drop=True)
    scripts.to_pickle("data/scripts.pkl")


# ===================================================


def encode_df_save(model):
    """this functions vectorizes reworked scripts and loads them to
    pickle file to be used as retrieval base for ranking script"""

    scripts_reopened = pd.read_pickle("data/scripts.pkl")
    vect_data = []
    for index, row in scripts_reopened.iterrows():
        vect = encode(
            texts=row["question"],
            model=model,
            intent=row["tag"],
            contexts=row["context"],
        )
        vect_data.append(vect)
    with open("data/scripts_vectors.pkl", "wb") as f:
        pickle.dump(vect_data, f)


# ===================================================


def top_candidates(score_lst_sorted, intent, initial_data, top=1):
    """this functions receives results of the cousine similarity ranking and
    returns top items' scores and their indices"""
    intent_idx = initial_data.index[initial_data["tag"] == intent]
    filtered_candiates = [item for item in score_lst_sorted if item[1][0] in intent_idx]
    scores = [item[0] for item in filtered_candiates]
    candidates_indexes = [item[1][0] for item in filtered_candiates]
    return scores[0:top], candidates_indexes[0:top]


# ===================================================


def candidates_reranking(
    top_candidates_idx_lst, conversational_history, utterance, initial_df, pipeline
):
    """this function applies trained bert classifier to identified candidates and
    returns their updated rank"""
    reranked_idx = {}
    for idx in top_candidates_idx_lst:

        combined_text = (
            " ".join(conversational_history)
            + " [SEP] "
            + utterance
            + " [SEP] "
            + initial_df.iloc[idx]["answer"]
        )

        prediction = pipeline(combined_text)
        if prediction[0]["label"] == "LABEL_0":
            reranked_idx[idx] = prediction[0]["score"]

    return reranked_idx


# ===================================================


def read_files_negative(path1, path2):
    """this functions creates training dataset for classifier incl negative
    examples and saves it to the pickle file"""

    star_wars = []
    for file in path1:
        star_wars.append(pd.read_csv(file, sep='"', on_bad_lines="warn"))
    total = pd.concat(star_wars, ignore_index=True)

    rick_and_morty = pd.read_csv(path2)
    negative_lines_to_add = list(rick_and_morty["line"])
    negative_lines_to_add.extend(list(total["dialogue"]))

    scripts_reopened = pd.read_pickle("data/scripts.pkl")
    scripts_reopened["label"] = 0
    source = random.sample(
        list(scripts_reopened[scripts_reopened["question"] != ""]["question"]), 7062
    )
    negative_lines_to_add.extend(source)
    random.shuffle(negative_lines_to_add)

    scripts_negative = scripts_reopened[["question", "context"]]
    scripts_negative["label"] = 1

    scripts_negative["answer"] = negative_lines_to_add[0 : len(scripts_negative)]

    fin_scripts = pd.concat([scripts_negative, scripts_reopened])

    fin_scripts = fin_scripts.sample(frac=1).reset_index(drop=True)
    fin_scripts["context"] = fin_scripts["context"].apply(lambda x: "".join(x))
    fin_scripts = fin_scripts[fin_scripts["question"] != ""]
    fin_scripts = fin_scripts[fin_scripts["answer"] != ""]
    fin_scripts["combined_all"] = (
        fin_scripts["context"]
        + "[SEP]"
        + fin_scripts["question"]
        + "[SEP]"
        + fin_scripts["answer"]
    )

    fin_scripts["combined_cq"] = (
        fin_scripts["context"] + "[SEP]" + fin_scripts["question"]
    )
    # fin_scripts = fin_scripts.dropna(how='any')
    fin_scripts.to_pickle("data/scripts_for_reranker.pkl")


# ===================================================


def intent_classification(question, answer, tag_model):
    greetings = ["hi", "hello", "greeting", "greetings", "hii", "helo", "hellow"]
    tokens = word_tokenize(answer.lower())
    for token in tokens:
        if token in greetings:
            return "greetings"
        else:
            intent = tag_model.predict_tag(question)
            return intent


# ===================================================


def change_names(sentences):
    lst_punct = string.punctuation
    lst_punct += "’"
    sheldon_friends = [
        "Penny",
        "Amy",
        "Leonard",
        "Stephanie",
        "Dr. Stephanie",
        "Raj",
        "Rebecca",
    ]
    tokens = word_tokenize(sentences)
    changes = "".join(
        "my friend" if i in sheldon_friends else i if i in lst_punct else f" {i}"
        for i in tokens
    ).strip()
    return changes


# ===================================================


def data_prep_biencoder(path1, path2):
    """this functions creates training dataset for classifier incl negative
    examples and saves it to the pickle file"""

    star_wars = []
    for file in path1:
        star_wars.append(pd.read_csv(file, sep='"', on_bad_lines="warn"))
    total = pd.concat(star_wars, ignore_index=True)

    rick_and_morty = pd.read_csv(path2)
    negative_lines_to_add = list(rick_and_morty["line"])
    negative_lines_to_add.extend(list(total["dialogue"]))

    scripts_reopened = pd.read_pickle("data/scripts.pkl")
    scripts_reopened["label"] = 0
    source = random.sample(
        list(scripts_reopened[scripts_reopened["question"] != ""]["question"]), 7062
    )
    negative_lines_to_add.extend(source)
    random.shuffle(negative_lines_to_add)

    scripts_negative = scripts_reopened[["question", "context", "answer"]]
    scripts_negative["label"] = 1

    scripts_negative["neg_answer"] = negative_lines_to_add[0 : len(scripts_negative)]

    fin_scripts = scripts_negative.sample(frac=1).reset_index(drop=True)
    fin_scripts["context"] = fin_scripts["context"].apply(lambda x: "".join(x))
    fin_scripts = fin_scripts[fin_scripts["question"] != ""]
    fin_scripts = fin_scripts[fin_scripts["answer"] != ""]

    fin_scripts["combined"] = fin_scripts["context"] + "[SEP]" + fin_scripts["question"]
    # fin_scripts = fin_scripts.dropna(how='any')
    fin_scripts.to_pickle("data/scripts_for_biencoder.pkl")
