import torch
import torch.nn.functional as F
from transformers import logging, AutoTokenizer
from transformers import AutoModelForNextSentencePrediction as AutoNSPModel

logging.set_verbosity_error()


class ModelManager:
    def __init__(
        self,
        model_name: str = "roberta-base",
        device = torch.device("cpu"),
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        self.cls_vector = self.get_cls_vector()
        self.sep_vector = self.get_sep_vector()

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        print(f"Finish load tokenizer - {self.model_name}")
        return tokenizer

    def load_model(self):
        model = AutoNSPModel.from_pretrained(self.model_name).to(self.device)
        print(f"Finish load model - {self.model_name}")
        return model

    def encode_sentences_with_sep(self, sentence_1: str, sentence_2: str):
        return self.tokenizer(sentence_1, sentence_2, return_tensors="pt").to(self.device)

    def get_cls_vector(self):
        cls_token = self.tokenizer(
            self.tokenizer.cls_token,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            cls_vector = self.model(**cls_token, output_hidden_states=True)
        return cls_vector.hidden_states[-1].squeeze(0)

    def get_sep_vector(self):
        sep_token = self.tokenizer(
            self.tokenizer.sep_token,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            sep_vector = self.model(**sep_token, output_hidden_states=True)
        return sep_vector.hidden_states[-1].squeeze(0)

    def get_nsp_score(self, sentence_1: str, sentence_2: str):
        encoded_sentences = self.encode_sentences_with_sep(sentence_1, sentence_2)
        with torch.no_grad():
            output = self.model(**encoded_sentences)
        output = F.softmax(output.logits, dim=1)
        return output

    def get_nsp_score_with_document(self, document_1: str, document_2: str):
        from libs.sentence_vector import generate_sentence_vectors

        sentence_vector_1 = generate_sentence_vectors(document_1, self)
        sentence_vector_2 = generate_sentence_vectors(document_2, self)
        # sentence_vector_1, sentence_vector_2: (# of sentences, embedding size)
        sentence_vectors = torch.cat(
            (self.cls_vector, sentence_vector_1, self.sep_vector, sentence_vector_2),
            dim=0,
        )
        # sentence_vectors: (total length, embedding size)
        if len(sentence_vectors) > 512:
            return None, True
        sentence_vectors = sentence_vectors.unsqueeze(0)
        # sentence_vectors: (1, total length, embedding size)

        with torch.no_grad():
            output = self.model(inputs_embeds=sentence_vectors)
        output = F.softmax(output.logits, dim=1)
        return output, False

    def get_nsp_score_with_sentence_vector(self, sentence_vector_1: torch.Tensor, sentence_vector_2: torch.Tensor):
        # sentence_vector_1, sentence_vector_2: (# of sentences, embedding size)
        sentence_vectors = torch.cat(
            (self.cls_vector, sentence_vector_1, self.sep_vector, sentence_vector_2),
            dim=0,
        )
        # sentence_vectors: (total length, embedding size)
        if len(sentence_vectors) > 512:
            return None, True
        sentence_vectors = sentence_vectors.unsqueeze(0)
        # sentence_vectors: (1, total length, embedding size)

        with torch.no_grad():
            output = self.model(inputs_embeds=sentence_vectors)
        output = F.softmax(output.logits, dim=1)
        return output, False


if __name__ == "__main__":
    manager = ModelManager(model_name="bert-base-uncased")

    # Test short sentence
    text1 = "Replace me by any text you'd like."
    text2 = "Last layer hidden-state of the first token of the sequence."
    output = manager.get_nsp_score(text1, text2)
    print(output)

    # Test long document
    doc1 = "Replace me by any text you'd like. Last layer hidden-state of the first token of the sequence."
    doc2 = "Last layer hidden-state of the first token of the sequence. Replace me by any text you'd like. Replace me by any text you'd like."
    output = manager.get_nsp_score_with_sentence_vector(doc1, doc2)
    print(output)
