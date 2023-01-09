from pathlib import Path

DATA_DIR = Path("/data/coherence_evaluation")
CACHE_DIR = DATA_DIR / "cache"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
