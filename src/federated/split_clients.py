from pathlib import Path
import shutil
import random

SOURCE_ROOT = Path("data/processed/covid_binary/train")
OUTPUT_ROOT = Path("data/processed/covid_clients")

NUM_CLIENTS = 4
SEED = 42
CLASSES = ["COVID", "Normal"]


def clear_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_list(items, num_parts):
    return [items[i::num_parts] for i in range(num_parts)]


def main():
    random.seed(SEED)
    clear_folder(OUTPUT_ROOT)

    for client_id in range(1, NUM_CLIENTS + 1):
        for cls in CLASSES:
            (OUTPUT_ROOT / f"client_{client_id}" / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        files = list((SOURCE_ROOT / cls).glob("*"))
        random.shuffle(files)

        splits = split_list(files, NUM_CLIENTS)

        for idx, split_files in enumerate(splits, start=1):
            dest = OUTPUT_ROOT / f"client_{idx}" / cls
            for file in split_files:
                shutil.copy2(file, dest / file.name)

    print("Created federated client datasets:")
    for client_id in range(1, NUM_CLIENTS + 1):
        print(f"\nclient_{client_id}")
        for cls in CLASSES:
            count = len(list((OUTPUT_ROOT / f"client_{client_id}" / cls).glob("*")))
            print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()