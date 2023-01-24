import json
import torch
from tqdm.auto import tqdm
from data_caching import CACHE_CAPTION_DIR, CACHE_PLOT_DIR
from libs.model import ModelManager
from libs.preprocess import load_data

DATA_DIR = "./data"
DATA_NAME = "desc_and_plot_train.csv"
OUTPUT_DIR = "./output"
device = torch.device("cuda:3")


def save_coherence_to_json(coherence_dict: dict):
    with open(f"{OUTPUT_DIR}/coherence.json", "w") as coherence_file:
        json.dump(coherence_dict, coherence_file, indent=2)


def get_score_result(row, manager: ModelManager, ground_truth: bool = True) -> dict:
    cached_plot_path = CACHE_PLOT_DIR / f"{row['movie_id']}.pt"
    cached_caption_path = CACHE_CAPTION_DIR / f"{row['video_id']}.pt"

    if cached_plot_path.exists() and cached_caption_path.exists():
        plot_vector = torch.load(cached_plot_path, map_location=device)
        caption_vector = torch.load(cached_caption_path, map_location=device)
        coherence_score, is_long_document = manager.get_nsp_score_with_sentence_vector(
            sentence_vector_1=plot_vector,
            sentence_vector_2=caption_vector,
        )
    else:
        coherence_score, is_long_document = manager.get_nsp_score_with_document(
            document_1=row["plot"],
            document_2=row["caption"],
        )
    if is_long_document:
        coherence_score = 0.0
    else:
        coherence_score = coherence_score[0][0].item()

    return {
        "movie_id": row["movie_id"],
        "target": 1.0 if ground_truth else 0.0,
        "predict": coherence_score,
    }


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
    manager = ModelManager(model_name="bert-base-uncased", device=device)

    # Calculate coherence between sentence
    coherence_dict = dict()
    for _, positive_row in tqdm(
        data_df.iterrows(),
        desc="Coherence",
        total=len(data_df),
    ):
        movie_id = positive_row["movie_id"]
        if movie_id not in coherence_dict:
            coherence_dict[movie_id] = dict()

        # Positive sample
        score_result = get_score_result(positive_row, manager, ground_truth=True)
        coherence_dict[positive_row["movie_id"]][
            positive_row["video_id"]
        ] = score_result

        # Negative sample
        while movie_id == positive_row["movie_id"]:
            negative_row = data_df.sample(n=1).iloc[0]
            movie_id = negative_row["movie_id"]
        score_result = get_score_result(negative_row, manager, ground_truth=False)
        coherence_dict[positive_row["movie_id"]][
            negative_row["video_id"]
        ] = score_result

    save_coherence_to_json(coherence_dict)


if __name__ == "__main__":
    main()
