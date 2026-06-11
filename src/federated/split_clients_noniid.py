from pathlib import Path
import shutil
import random

SOURCE_ROOT = Path("data/processed/covid_binary/train")
OUTPUT_ROOT = Path("data/processed/covid_clients_noniid")

SEED = 42

CLIENT_DISTRIBUTION = {
    "client_1": {"COVID": 320, "Normal": 80},
    "client_2": {"COVID": 280, "Normal": 120},
    "client_3": {"COVID": 120, "Normal": 280},
    "client_4": {"COVID": 80, "Normal": 320},
}


def clear_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    random.seed(SEED)
    clear_folder(OUTPUT_ROOT)

    available_files = {}

    for cls in ["COVID", "Normal"]:
        files = list((SOURCE_ROOT / cls).glob("*"))
        random.shuffle(files)
        available_files[cls] = files

    used_index = {"COVID": 0, "Normal": 0}

    for client, class_counts in CLIENT_DISTRIBUTION.items():
        for cls, count in class_counts.items():
            dest = OUTPUT_ROOT / client / cls
            dest.mkdir(parents=True, exist_ok=True)

            start = used_index[cls]
            end = start + count
            selected_files = available_files[cls][start:end]

            if len(selected_files) < count:
                raise ValueError(f"Not enough files for {client} - {cls}")

            for file in selected_files:
                shutil.copy2(file, dest / file.name)

            used_index[cls] = end

    print("Created Non-IID federated client datasets:")
    for client in CLIENT_DISTRIBUTION:
        print(f"\n{client}")
        for cls in ["COVID", "Normal"]:
            count = len(list((OUTPUT_ROOT / client / cls).glob("*")))
            print(f"  {cls}: {count}")

    print("\nTotal used:")
    for cls in ["COVID", "Normal"]:
        print(f"  {cls}: {used_index[cls]}")


if __name__ == "__main__":
    main()