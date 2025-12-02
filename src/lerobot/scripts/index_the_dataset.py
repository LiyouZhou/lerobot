from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import csv
from tqdm import main, tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main(repo_id, repo_root, output_csv):
    # ds_meta = LeRobotDatasetMetadata(repo_id, root=repo_root)

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=repo_root,
        video_backend="torchcodec",
    )

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["frame_index", "episode_index", "index", "task_index", "task_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample in tqdm(dataset):
            sample_info = {}
            for key in fieldnames:
                try:
                    sample_info[key] = int(sample[key].cpu().numpy())
                except Exception as e:
                    sample_info[key] = str(sample[key])
            writer.writerow(sample_info)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Index the dataset and save to a CSV file."
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default="/home/liyouzhou/study/any4lerobot/openx2lerobot/data/mikasa_robo_tfds_all_1.0.0_lerobot",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    repo_id = repo_root.name
    output_csv = repo_root / "dataset_index.csv"

    main(repo_id=repo_id, repo_root=repo_root, output_csv=output_csv)
