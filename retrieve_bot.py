import pandas as pd
import pickle
import random
from sentence_transformers import SentenceTransformer
from utils import (
    encode,
    cosine_sim,
    top_candidates,
    candidates_reranking,
    intent_classification,
)
from collections import deque
from transformers import pipeline
import torch
import json
from transformers import AutoTokenizer
from dialog_tag import DialogTag

# this class representes main functions of retrieve bot


class ChatBot:
    def __init__(self):
        self.vect_data = []
        self.scripts = []
        self.conversation_history = deque([], maxlen=5)
        self.tag_model = None
        self.ranking_model = None
        self.reranking_model = None
        self.device = None
        self.tokenizer = None
        self.low_scoring_list = None

    def load(self):
        """ "This method is called first to load all datasets and
        model used by the chat bot; all the data to be saved in
        tha data folder, models to be loaded from hugging face"""

        with open("data/scripts_vectors.pkl", "rb") as fp:
            self.vect_data = pickle.load(fp)
        self.scripts = pd.read_pickle("data/scripts.pkl")
        with open("data/low_score_sripts.json", "r") as f:
            self.low_scoring_list = json.load(f)
        self.tag_model = DialogTag("distilbert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ranking_model = SentenceTransformer(
            "Shakhovak/chatbot_sentence-transformer"
        )  # # sentence-transformers/LaBSE  or sentence-transformers/all-mpnet-base-v2 or Shakhovak/chatbot_sentence-transformer

        self.tokenizer_reranking = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.reranking_model = pipeline(
            model="Shakhovak/RerankerModel_chat_bot",
            device=self.device,
            tokenizer=self.tokenizer_reranking,
        )

    def generate_response(self, utterance: str) -> str:
        """this functions identifies potential
        candidates for answer and ranks them"""

        intent = intent_classification(utterance, utterance, self.tag_model)
        query_encoding = encode(
            texts=utterance,
            intent=intent,
            model=self.ranking_model,
            contexts=self.conversation_history,
        )
        bot_cosine_scores = cosine_sim(
            self.vect_data,
            query_encoding,
        )
        top_scores, top_indexes = top_candidates(
            bot_cosine_scores, intent=intent, initial_data=self.scripts, top=5
        )
        if top_scores[0] < 0.9:
            if intent == "greetings":
                answer = random.choice(self.low_scoring_list["greetings"])
                self.conversation_history.clear()
            else:
                answer = random.choice(self.low_scoring_list["generic"])
                self.conversation_history.clear()
        else:
            # test candidates and collects them with label 0 to dictionary

            reranked_dict = candidates_reranking(
                top_indexes,
                self.conversation_history,
                utterance,
                self.scripts,
                self.reranking_model,
            )
            # if any candidates were selected, range them and pick up the top
            # else keep up the initial top 1

            if len(reranked_dict) >= 1:
                updated_top_candidates = dict(
                    sorted(reranked_dict.items(), key=lambda item: item[1])
                )
                answer = self.scripts.iloc[list(updated_top_candidates.keys())[0]][
                    "answer"
                ]
            else:
                answer = self.scripts.iloc[top_indexes[0]]["answer"]

        self.conversation_history.append(utterance)
        self.conversation_history.append(answer)

        return answer


# katya = ChatBot()
# katya.load()
# katya.generate_response("hi man!")
