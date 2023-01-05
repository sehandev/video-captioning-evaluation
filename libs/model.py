import torch
import torch.nn.functional as F
from transformers import logging, AutoTokenizer, AutoModelForNextSentencePrediction

logging.set_verbosity_error()


class ModelManager:
    def __init__(self, model_name: str = "roberta-base"):
        self.model_name = model_name
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        print(f"Finish load tokenizer - {self.model_name}")
        return tokenizer

    def load_model(self):
        model = AutoModelForNextSentencePrediction.from_pretrained(self.model_name).to("cuda")
        print(f"Finish load model - {self.model_name}")
        return model

    def encode_sentences_with_sep(self, sentence_1: str, sentence_2: str):
        return self.tokenizer(sentence_1, sentence_2, return_tensors="pt")

    def get_nsp_score(self, sentence_1: str, sentence_2: str):
        encoded_sentences = self.encode_sentences_with_sep(sentence_1, sentence_2)
        with torch.no_grad():
            output = self.model(**encoded_sentences)
        output = F.softmax(output.logits, dim=1)
        return output


if __name__ == "__main__":
    manager = ModelManager(model_name="bert-base-uncased")
    text1 = "Replace me by any text you'd like."
    text2 = "Last layer hidden-state of the first token of the sequence."
    output = manager.get_nsp_score(text1, text2)
    print(output)
