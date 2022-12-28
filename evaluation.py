import json
from tqdm.auto import tqdm
from libs.model import ModelManager
from libs.preprocess import load_data

DATA_DIR = "./data"
DATA_NAME = "descriptions_train_val.csv"
OUTPUT_DIR = "./output"


def save_coherence_to_json(coherence_dict: dict):
    with open(f"{OUTPUT_DIR}/coherence.json", "w") as coherence_file:
        json.dump(coherence_dict, coherence_file, indent=2)


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
    manager = ModelManager(model_name="bert-base-uncased")

    # Calculate coherence between sentence
    coherence_dict = dict()
    for _, front_row in tqdm(
        data_df.iterrows(),
        desc="Coherence",
        total=len(data_df),
        position=0,
    ):
        coherence_dict[front_row.clip_id] = dict()
        front_sentence = front_row.caption
        for _, back_row in tqdm(
            data_df.iterrows(),
            total=len(data_df),
            position=1,
            leave=False,
        ):
            back_sentence = back_row.caption
            coherence_score = manager.get_nsp_score(front_sentence, back_sentence)
            coherence_score = coherence_score[0][0].item()
            coherence_dict[front_row.clip_id][back_row.clip_id] = coherence_score

    save_coherence_to_json(coherence_dict)


if __name__ == "__main__":
    main()
