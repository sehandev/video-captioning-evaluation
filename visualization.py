import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path("./output")
RESULT_PATH = OUTPUT_DIR / "coherence.json"
IMAGE_PATH = OUTPUT_DIR / "coherence.jpg"


def main():
    with open(RESULT_PATH, "r") as result_file:
        result_dict = json.load(result_file)

    new_dict = {
        "predict": [],
        "target": [],
    }

    for score_dict in result_dict.values():
        for tmp_dict in score_dict.values():
            new_dict["predict"].append(tmp_dict["predict"])
            new_dict["target"].append(bool(tmp_dict["target"]))

    score_df = pd.DataFrame.from_dict(new_dict)

    sns.stripplot(
        data=score_df,
        x="target",
        y="predict",
        hue="target",
        legend=False,
    )

    plt.title("Coherence score range")
    plt.savefig(IMAGE_PATH)


if __name__ == "__main__":
    main()
