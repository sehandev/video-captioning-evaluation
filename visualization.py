import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("./output")
RESULT_PATH = OUTPUT_DIR / "coherence.json"
IMAGE_PATH = OUTPUT_DIR / "coherence.jpg"


def main():
    with open(RESULT_PATH, "r") as result_file:
        result_dict = json.load(result_file)

    coherence_score_list = list(result_dict.values())

    score_range_dict = {
        "0.0": 0,
        "0.1": 0,
        "0.2": 0,
        "0.3": 0,
        "0.4": 0,
        "0.5": 0,
        "0.6": 0,
        "0.7": 0,
        "0.8": 0,
        "0.9": 0,
        "1.0": 0,
    }
    for coherence_score in coherence_score_list:
        if coherence_score == 0.0:
            score_range_dict["0.0"] += 1
        elif coherence_score < 0.1:
            score_range_dict["0.1"] += 1
        elif coherence_score < 0.2:
            score_range_dict["0.2"] += 1
        elif coherence_score < 0.3:
            score_range_dict["0.3"] += 1
        elif coherence_score < 0.4:
            score_range_dict["0.4"] += 1
        elif coherence_score < 0.5:
            score_range_dict["0.5"] += 1
        elif coherence_score < 0.6:
            score_range_dict["0.6"] += 1
        elif coherence_score < 0.7:
            score_range_dict["0.7"] += 1
        elif coherence_score < 0.8:
            score_range_dict["0.8"] += 1
        elif coherence_score < 0.9:
            score_range_dict["0.9"] += 1
        else:
            score_range_dict["1.0"] += 1

    sns.set_theme(style="dark")
    sns.scatterplot(data=score_range_dict)
    
    plt.title('Coherence score range')
    plt.savefig(IMAGE_PATH)


if __name__ == "__main__":
    main()
