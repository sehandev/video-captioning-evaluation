from transformers import logging, AutoTokenizer, AutoModel

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
        model = AutoModel.from_pretrained(self.model_name)
        print(f"Finish load model - {self.model_name}")
        return model

    def encode_sentence(self, sentence: str):
        return self.tokenizer(sentence, return_tensors="pt")

    def get_cls_vector(self, sentence: str):
        encoded_sentence = self.encode_sentence(sentence)
        output = self.model(**encoded_sentence)
        output = output.pooler_output
        return output


if __name__ == "__main__":
    manager = ModelManager(model_name="roberta-base")
    text = "Replace me by any text you'd like."
    output = manager.get_cls_vector(text)
    print(output.shape)
