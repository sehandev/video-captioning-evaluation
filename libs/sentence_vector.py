from typing import List
import torch
from libs.model import ModelManager


def split_document(document: str) -> List[str]:
    sentence_list = document.split(".")
    sentence_list = [sentence for sentence in sentence_list if len(sentence) > 1]
    return sentence_list


def generate_sentence_vectors(document: str, manager: ModelManager):
    sentence_list = split_document(document)
    encoded_sentences = manager.tokenizer(
        sentence_list,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(manager.device)

    with torch.no_grad():
        output = manager.model(**encoded_sentences, output_hidden_states=True)
        # output: (# of sentences, # of word, embedding size)
    output = torch.mean(output.hidden_states[-1], dim=1)
    # output: (# of sentences, embedding size)

    return output


if __name__ == "__main__":
    manager = ModelManager(model_name="bert-base-uncased")
    sample_document = "Replace me by any text you'd like. Last layer hidden-state of the first token of the sequence."
    output = generate_sentence_vectors(sample_document, manager)
    print(output.shape)
