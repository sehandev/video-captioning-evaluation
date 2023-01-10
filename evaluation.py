import json
from tqdm.auto import tqdm
from libs.model import ModelManager
from libs.preprocess import load_data

DATA_DIR = "./data"
DATA_NAME = "desc_and_plot_train.csv"
OUTPUT_DIR = "./output"


def save_coherence_to_json(coherence_dict: dict):
    with open(f"{OUTPUT_DIR}/coherence.json", "w") as coherence_file:
        json.dump(coherence_dict, coherence_file, indent=2)


def main():
    data_df = load_data(
        data_dir=DATA_DIR,
        data_filename=DATA_NAME,
        rename_dict={
            "videoid": "video_id",
            "imdbid": "movie_id",
            "caption_from_old": "caption",
            "synopsis": "plot",
        },
    )
    manager = ModelManager(model_name="bert-base-uncased")

    # Calculate coherence between sentence
    coherence_dict = dict()
    for _, row in tqdm(
        data_df.iterrows(),
        desc="Coherence",
        total=len(data_df),
    ):
        # TODO plot 1개에 해당하는 caption n개, 다른 caption n개를 NSP
        # 현재 plot 1개, 해당하는 caption n개, 다른 caption 0개
        coherence_dict[row["video_id"]] = dict()
        coherence_score, is_long_document = manager.get_nsp_score_with_sentence_vector(
            row["plot"], row["caption"]
        )
        if is_long_document:
            coherence_score = 0.0
        else:
            coherence_score = coherence_score[0][0].item()
        coherence_dict[row["video_id"]] = coherence_score

    save_coherence_to_json(coherence_dict)


if __name__ == "__main__":
    main()
