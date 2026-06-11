from pathlib import Path
import shutil
import random

RAW_ROOT = Path("data/raw/covid/covid_radiography")
OUT_ROOT = Path("data/processed/covid_binary")

CLASSES = {
    "COVID": RAW_ROOT / "COVID" / "images",
    "Normal": RAW_ROOT / "Normal" / "images",
}

TRAIN_PER_CLASS = 800
TEST_PER_CLASS = 200
SEED = 42


def clear_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_images(files, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, destination / file.name)


def main():
    random.seed(SEED)

    clear_folder(OUT_ROOT)

    for split in ["train", "test"]:
        for cls in CLASSES:
            (OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

    for cls, src_dir in CLASSES.items():
        images = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg"))
        random.shuffle(images)

        needed = TRAIN_PER_CLASS + TEST_PER_CLASS
        if len(images) < needed:
            raise ValueError(f"Not enough images for {cls}. Found {len(images)}, need {needed}")

        selected = images[:needed]
        train_files = selected[:TRAIN_PER_CLASS]
        test_files = selected[TRAIN_PER_CLASS:needed]

        copy_images(train_files, OUT_ROOT / "train" / cls)
        copy_images(test_files, OUT_ROOT / "test" / cls)

        print(f"{cls}: train={len(train_files)}, test={len(test_files)}")

    print("\nPrepared dataset at:", OUT_ROOT)


if __name__ == "__main__":
    main()