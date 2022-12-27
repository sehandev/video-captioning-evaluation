import torch
from tqdm.auto import tqdm
from libs.model import ModelManager
from libs.preprocess import load_data

DATA_DIR = "./data"
DATA_NAME = "descriptions_train_val.csv"


def calculate_similarity(sentence1: str, sentence2: str) -> float:
    return 0.0


def main():
    data_df = load_data(
        data_dir=DATA_DIR,
        data_filename=DATA_NAME,
        rename_dict={
            "videoid": "clip_id",
            "imdbid": "video_id",
            "caption_from_old": "caption",
        },
    )
    manager = ModelManager(model_name="roberta-base")
    for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
        cls_vector = manager.get_cls_vector(row["caption"])
        torch.save(cls_vector, f"{DATA_DIR}/cls/{row.clip_id}.pt")

    # similarity = calculate_similarity("abc", "bcd")
    # print(similarity)


if __name__ == "__main__":
    main()
