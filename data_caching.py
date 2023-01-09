from pathlib import Path
from libs.preprocess import load_data

DATA_DIR = Path("/data/coherence_evaluation")
CACHE_DIR = DATA_DIR / "cache"
DATA_NAME = "desc_and_plot_train_val.csv"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
    print(data_df)


if __name__ == "__main__":
    main()
