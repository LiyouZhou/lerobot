#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterator
from pathlib import Path
import pandas as pd
import numpy as np

from torch.utils.data import Sampler
import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(
                        start_index.item() + drop_n_first_frames,
                        end_index.item() - drop_n_last_frames,
                    )
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class EpisodicBatchSampler(Sampler):
    def __init__(self, repo_root, batch_size, shuffle=True, allowable_task_names=None):
        self.repo_root = repo_root
        self.dataset_index_df = pd.read_csv(Path(repo_root) / "dataset_index.csv")

        self.episode_counts = []
        for task_index in self.dataset_index_df["task_index"].unique():
            task_df = self.dataset_index_df[
                self.dataset_index_df["task_index"] == task_index
            ]
            episode_count = len(task_df["episode_index"].value_counts())

            max_frame_index = task_df["frame_index"].max()
            episode_length = max_frame_index + 1

            self.episode_counts.append((task_index, episode_count, episode_length))

        # Set the allowable task indices
        self.allowable_task_names = self.dataset_index_df["task_name"].unique().tolist()
        if allowable_task_names is not None:
            assert isinstance(allowable_task_names, list), "allowable_task_names should be a list"
            for name in allowable_task_names:
                assert name in self.allowable_task_names, f"Task name {name} is not in the dataset"
            self.allowable_task_names = allowable_task_names

        # Get a list of episodes with allowable task indices
        self.allowable_episodes_indices = self.dataset_index_df[
            self.dataset_index_df["task_name"].isin(self.allowable_task_names)
        ]["episode_index"].unique()

        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self.dataset_index_df) // self.batch_size):
            # sample a batch of episodes
            sampled_episode_indices = np.random.choice(
                self.allowable_episodes_indices,
                self.batch_size,
            )

            # Get all rows corresponding to the sampled episode
            sampled_episodes_df = self.dataset_index_df[
                self.dataset_index_df["episode_index"].isin(sampled_episode_indices)
            ]

            # Count how many rows in the dataframe correspond to each episode
            episode_lengths = sampled_episodes_df["episode_index"].value_counts()
            max_episode_length = episode_lengths.max()

            for frame_index in range(max_episode_length):
                all_indices = []
                for sampled_episode_index in sampled_episode_indices:
                    sampled_frame_index = (
                        frame_index
                        if frame_index < episode_lengths[sampled_episode_index]
                        else episode_lengths[sampled_episode_index] - 1
                    )
                    # Get all indices for the sampled episode within the
                    # sampled task at the current frame index
                    sample = sampled_episodes_df[
                        (sampled_episodes_df["episode_index"] == sampled_episode_index)
                        & (sampled_episodes_df["frame_index"] == sampled_frame_index)
                    ]
                    assert (
                        len(sample) == 1
                    ), f"Expected one sample for episode {sampled_episode_index} at frame {frame_index}, got {len(sample)}"
                    all_indices.extend(sample["index"].tolist())
                yield all_indices

    def __len__(self):
        return len(self.dataset_index_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root", type=str, required=True, help="Path to the repository root"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for sampling"
    )
    parser.add_argument(
        "--length", type=int, default=20, help="Number of batches to sample"
    )

    args = parser.parse_args()

    # Example usage
    sampler = EpisodicBatchSampler(
        repo_root=args.repo_root,
        batch_size=args.batch_size,
        shuffle=True,
        remember_color_only=False,
    )

    for i, batch in enumerate(sampler):
        print(f"Batch {i}: {batch}")
        if i >= args.length:
            break
