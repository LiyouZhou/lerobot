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
    def __init__(self, repo_root, batch_size, shuffle=True, remember_color_only=False):
        self.repo_root = repo_root
        self.dataset_index_df = pd.read_csv(Path(repo_root) / "dataset_index.csv")

        self.task_info = {}
        self.episode_counts = []
        for task_index in self.dataset_index_df["task_index"].unique():
            task_df = self.dataset_index_df[
                self.dataset_index_df["task_index"] == task_index
            ]
            episode_count = len(task_df["episode_index"].value_counts())

            max_frame_index = task_df["frame_index"].max()
            episode_length = max_frame_index + 1

            self.episode_counts.append((task_index, episode_count, episode_length))
            self.task_info[task_index] = {
                "episode_count": episode_count,
                "episode_length": episode_length,
            }

            assert (
                task_df.groupby("episode_index")["frame_index"].nunique().nunique() == 1
            ), f"Task {task_index} has episodes of varying lengths"

        # Extract task indices and their episode counts
        self.task_indices = [t[0] for t in self.episode_counts]
        self.episode_counts_list = [t[1] for t in self.episode_counts]

        self.task_probabilities = np.array(self.episode_counts_list) / np.sum(self.episode_counts_list)
        # ('5', 'touch the red cube')
        # ('11', 'touch the maroon cube')
        # ('21', 'touch the orange cube')
        # ('16', 'touch the yellow cube')
        # ('1', 'Memorize the the colors of the cube shown on the table, and then touch the same coloured cube out of all the cubes.')
        # ('18', 'touch the purple cube')
        # ('20', 'touch the green cube')
        # ('19', 'touch the cyan cube')
        # ('8', 'touch the teal cube')
        # ('17', 'touch the blue cube')
        remember_color_indices = [5, 11, 21, 16, 1, 18, 20, 19, 8, 17]
        if remember_color_only:
            # Set probability of non-remember_color tasks to 0
            for i, task_index in enumerate(self.task_indices):
                if task_index not in remember_color_indices:
                    self.task_probabilities[i] = 0
            # Re-normalize probabilities
            self.task_probabilities /= np.sum(self.task_probabilities)

        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self.dataset_index_df) // self.batch_size):
            # Sample a task index with probability proportional to episode count
            sampled_task_index = np.random.choice(
                self.task_indices,
                p=self.task_probabilities,
            )

            # Filter the dataframe for the sampled task index
            task_df = self.dataset_index_df[
                self.dataset_index_df["task_index"] == sampled_task_index
            ]

            # Get all unique episode indices for this task
            episode_indices = task_df["episode_index"].unique()

            # Sample batch_size episodes index uniformly
            sampled_episode_indices = np.random.choice(
                episode_indices, size=self.batch_size, replace=True
            )

            episode_length = self.task_info[sampled_task_index]["episode_length"]
            for frame_index in range(1, episode_length):
                all_indices = []
                for sampled_episode_index in sampled_episode_indices:
                    # Get all indices for the sampled episode within the 
                    # sampled task at the current frame index
                    episode_indices_df = task_df[
                        task_df["episode_index"] == sampled_episode_index
                    ]
                    sample = episode_indices_df[
                        episode_indices_df["frame_index"] == frame_index
                    ]
                    assert (
                        len(sample) == 1
                    ), f"Expected one sample for episode {sampled_episode_index} at frame {frame_index}, got {len(sample)}"
                    all_indices.extend(sample["index"].tolist())
                yield all_indices

    def __len__(self):
        return len(self.dataset_index_df)
