from pathlib import Path
import torch
from tqdm.auto import tqdm
from libs.model import ModelManager
from libs.preprocess import load_data
from libs.sentence_vector import generate_sentence_vectors

DATA_DIR = Path("/data/coherence_evaluation")
CACHE_DIR = DATA_DIR / "cache"
CACHE_PLOT_DIR = CACHE_DIR / "plot"
CACHE_CAPTION_DIR = CACHE_DIR / "caption"
DATA_NAME = "desc_and_plot_train_val.csv"


def main(
    model_name: str = "bert-base-uncased",
):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_CAPTION_DIR.mkdir(parents=True, exist_ok=True)

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

    manager = ModelManager(
        model_name,
        device=torch.device("cuda:0"),
    )
    torch.save(manager.cls_vector, CACHE_DIR / "cls.pt")
    torch.save(manager.sep_vector, CACHE_DIR / "sep.pt")

    for _, row in tqdm(
        data_df.iterrows(),
        desc="Caching",
        total=len(data_df),
    ):
        # Plot
        vector_path = CACHE_PLOT_DIR / f"{row['movie_id']}.pt"
        if not vector_path.exists():
            plot_vector = generate_sentence_vectors(row["plot"], manager)
            torch.save(plot_vector, vector_path)

        # Caption
        vector_path = CACHE_CAPTION_DIR / f"{row['video_id']}.pt"
        if not vector_path.exists():
            caption_vector = generate_sentence_vectors(row["caption"], manager)
            torch.save(caption_vector, vector_path)


if __name__ == "__main__":
    main(model_name="bert-base-uncased")
