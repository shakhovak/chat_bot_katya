import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from utils import encode, cosine_sim, top_candidates, candidates_reranking
from collections import deque
from transformers import pipeline
import torch
from transformers import AutoTokenizer

# this class representes main functions of retrieve bot


class ChatBot:
    def __init__(self):
        self.vect_data = []
        self.scripts = []
        self.conversation_history = deque([], maxlen=5)
        self.ranking_model = None
        self.reranking_model = None
        self.device = None
        self.tokenizer = None

    def load(self):
        """ "This method is called first to load all datasets and
        model used by the chat bot; all the data to be saved in
        tha data folder, models to be loaded from hugging face"""

        with open("data/scripts_vectors.pkl", "rb") as fp:
            self.vect_data = pickle.load(fp)
            self.scripts = pd.read_pickle("data/scripts.pkl")
        self.ranking_model = SentenceTransformer("sentence-transformers/LaBSE")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.reranking_model = pipeline(
            model="Shakhovak/RerankerModel_chat_bot",
            device=self.device,
            tokenizer=self.tokenizer,
        )

    def generate_response(self, utterance: str) -> str:
        """this functions identifies potential
        candidates for answer and ranks them"""
        query_encoding = encode(
            utterance, self.ranking_model, contexts=self.conversation_history
        )
        bot_cosine_scores = cosine_sim(self.vect_data, query_encoding)
        top_scores, top_indexes = top_candidates(bot_cosine_scores, top=20)

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
            answer = self.scripts.iloc[list(updated_top_candidates.keys())[0]]["answer"]
        else:
            answer = self.scripts.iloc[top_indexes[0]]["answer"]

        self.conversation_history.append(utterance)
        self.conversation_history.append(answer)

        return answer
