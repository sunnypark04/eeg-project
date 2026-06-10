from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Sequence

import mne
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGERY_LEFT_RIGHT_RUNS = (4, 8, 12)

LABEL_TO_INDEX = {
    "left_fist_imagery": 0,
    "right_fist_imagery": 1,
}

INDEX_TO_LABEL = {
    0: "left_fist_imagery",
    1: "right_fist_imagery",
}

ANNOTATION_TO_LABEL = {
    "T1": "left_fist_imagery",
    "T2": "right_fist_imagery",
}

MOTOR_21_CHANNELS = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
]

CENTRAL_3_CHANNELS = ["C3", "CZ", "C4"]


def clean_channel_name(name: str) -> str:
    return (
        name.upper()
        .replace(".", "")
        .replace(" ", "")
        .replace("-", "")
        .strip()
    )


def parse_subject_and_run(file_path: Path) -> tuple[int, int]:
    match = re.search(r"S(\d{3})R(\d{2})", file_path.name.upper())
    if match is None:
        raise ValueError(f"Could not parse subject/run from filename: {file_path.name}")

    subject = int(match.group(1))
    run = int(match.group(2))
    return subject, run


def discover_edf_files(
    data_dir: str | Path,
    runs: Sequence[int] = IMAGERY_LEFT_RIGHT_RUNS,
    subjects: Optional[Sequence[int]] = None,
) -> list[Path]:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    wanted_runs = set(runs)
    wanted_subjects = set(subjects) if subjects is not None else None

    files = []

    for path in sorted(data_dir.rglob("*.edf")):
        try:
            subject, run = parse_subject_and_run(path)
        except ValueError:
            continue

        if run not in wanted_runs:
            continue

        if wanted_subjects is not None and subject not in wanted_subjects:
            continue

        files.append(path)

    return files


def discover_subjects(
    data_dir: str | Path,
    runs: Sequence[int] = IMAGERY_LEFT_RIGHT_RUNS,
) -> list[int]:
    files = discover_edf_files(data_dir=data_dir, runs=runs)
    return sorted({parse_subject_and_run(path)[0] for path in files})


def make_subject_splits(
    subjects: Sequence[int],
    seed: int = 42,
    train_frac: float = 0.65,
    val_frac: float = 0.17,
) -> tuple[list[int], list[int], list[int]]:
    subjects = sorted(set(subjects))

    if len(subjects) < 3:
        raise ValueError(
            "Need at least 3 subjects for train/val/test split. "
            f"Found: {subjects}"
        )

    rng = np.random.default_rng(seed)
    subjects_array = np.array(subjects)
    rng.shuffle(subjects_array)

    n = len(subjects_array)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))

    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    train_subjects = sorted(subjects_array[:n_train].tolist())
    val_subjects = sorted(subjects_array[n_train:n_train + n_val].tolist())
    test_subjects = sorted(subjects_array[n_train + n_val:].tolist())

    return train_subjects, val_subjects, test_subjects


def read_raw_edf(
    file_path: Path,
    target_sfreq: float = 160.0,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    try:
        mne.datasets.eegbci.standardize(raw)
    except Exception:
        pass

    raw.pick_types(eeg=True, exclude=[])

    current_sfreq = float(raw.info["sfreq"])

    if target_sfreq is not None and abs(current_sfreq - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, npad="auto", verbose=False)

    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    return raw


def get_requested_channels(
    raw,
    channel_set: str,
    custom_channels: Optional[Sequence[str]] = None,
) -> list[str]:
    channel_set = channel_set.lower()

    if channel_set == "full":
        return list(raw.ch_names)

    if channel_set in {"motor21", "motor", "reduced"}:
        return MOTOR_21_CHANNELS

    if channel_set in {"central3", "central"}:
        return CENTRAL_3_CHANNELS

    if channel_set == "custom":
        if custom_channels is None:
            raise ValueError("For channel_set='custom', provide custom_channels.")
        return list(custom_channels)

    raise ValueError("channel_set must be one of: full, motor21, central3, custom")


def get_channel_indices(
    raw,
    channel_set: str,
    custom_channels: Optional[Sequence[str]] = None,
) -> tuple[list[int], list[str]]:
    requested = get_requested_channels(raw, channel_set, custom_channels)

    lookup = {
        clean_channel_name(name): idx
        for idx, name in enumerate(raw.ch_names)
    }

    indices = []
    found_names = []
    missing = []

    for name in requested:
        key = clean_channel_name(name)

        if key in lookup:
            idx = lookup[key]
            indices.append(idx)
            found_names.append(raw.ch_names[idx])
        else:
            missing.append(name)

    if missing:
        raise ValueError(
            f"Missing requested channels: {missing}\n"
            f"Available channels include: {raw.ch_names[:20]}"
        )

    return indices, found_names


class EEGMotorImageryDataset(Dataset):
    """
    Event-window dataset for PhysioNet EEG Motor Movement/Imagery.

    Task:
        imagined left fist vs imagined right fist

    Runs:
        R04, R08, R12

    Labels:
        T1 -> 0 -> left_fist_imagery
        T2 -> 1 -> right_fist_imagery

    Output:
        x: [channels, time_points]
        y: integer class label
    """

    def __init__(
        self,
        data_dir: str | Path,
        subjects: Optional[Sequence[int]] = None,
        runs: Sequence[int] = IMAGERY_LEFT_RIGHT_RUNS,
        channel_set: str = "full",
        custom_channels: Optional[Sequence[str]] = None,
        tmin: float = 0.5,
        tmax: float = 3.5,
        normalize: bool = True,
        target_sfreq: float = 160.0,
        l_freq: float = 1.0,
        h_freq: float = 40.0,
    ):
        self.data_dir = Path(data_dir)
        self.subjects = list(subjects) if subjects is not None else None
        self.runs = tuple(runs)
        self.channel_set = channel_set
        self.custom_channels = list(custom_channels) if custom_channels is not None else None
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.normalize = normalize
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq

        if self.tmax <= self.tmin:
            raise ValueError("tmax must be greater than tmin.")

        self.files = discover_edf_files(
            data_dir=self.data_dir,
            runs=self.runs,
            subjects=self.subjects,
        )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No matching EDF files found in {self.data_dir}. "
                "Expected files like S001R04.edf, S001R08.edf, S001R12.edf."
            )

        self.samples = []
        self.sfreq = None
        self.channel_names = None
        self._raw_cache = {}

        self._build_index()

    def _build_index(self) -> None:
        for file_path in self.files:
            subject, run = parse_subject_and_run(file_path)
            raw = read_raw_edf(
                file_path,
                target_sfreq=self.target_sfreq,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
            )
            if self.sfreq is None:
                self.sfreq = float(raw.info["sfreq"])

            _, selected_names = get_channel_indices(
                raw,
                channel_set=self.channel_set,
                custom_channels=self.custom_channels,
            )

            if self.channel_names is None:
                self.channel_names = selected_names

            try:
                events, found_event_id = mne.events_from_annotations(
                    raw,
                    event_id={"T1": 1, "T2": 2},
                    verbose=False,
                )
            except ValueError:
                continue

            id_to_annotation = {v: k for k, v in found_event_id.items()}

            sfreq = float(raw.info["sfreq"])
            start_offset = int(round(self.tmin * sfreq))
            stop_offset = int(round(self.tmax * sfreq))

            for event in events:
                onset = int(event[0])
                event_code = int(event[2])

                annotation = id_to_annotation.get(event_code)

                if annotation not in ANNOTATION_TO_LABEL:
                    continue

                label_name = ANNOTATION_TO_LABEL[annotation]
                label = LABEL_TO_INDEX[label_name]

                start = onset + start_offset
                stop = onset + stop_offset

                if start < 0 or stop > raw.n_times:
                    continue

                self.samples.append(
                    {
                        "file_path": file_path,
                        "subject": subject,
                        "run": run,
                        "annotation": annotation,
                        "label": label,
                        "label_name": label_name,
                        "start": start,
                        "stop": stop,
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "No event windows were created. Check that the data contains "
                "R04/R08/R12 files with T1/T2 annotations."
            )

    def _get_raw(self, file_path: Path):
        file_path = Path(file_path)

        if file_path not in self._raw_cache:
         self._raw_cache[file_path] = read_raw_edf(
            file_path,
            target_sfreq=self.target_sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
        )

        return self._raw_cache[file_path]

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    @property
    def num_timepoints(self) -> int:
        return int(round((self.tmax - self.tmin) * self.sfreq))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        raw = self._get_raw(item["file_path"])

        channel_indices, _ = get_channel_indices(
            raw,
            channel_set=self.channel_set,
            custom_channels=self.custom_channels,
        )

        x = raw.get_data(
            picks=channel_indices,
            start=item["start"],
            stop=item["stop"],
        ).astype(np.float32)

        expected_timepoints = self.num_timepoints

        if x.shape[1] > expected_timepoints:
            x = x[:, :expected_timepoints]

        elif x.shape[1] < expected_timepoints:
            pad_width = expected_timepoints - x.shape[1]
            x = np.pad(
                x,
                pad_width=((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=0,
            )

        if self.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mean) / (std + 1e-6)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(item["label"], dtype=torch.long)

        return x, y

    def get_metadata(self, idx: int) -> dict:
        item = dict(self.samples[idx])
        item["file_path"] = str(item["file_path"])
        return item

    def summary(self) -> dict:
        labels = [sample["label_name"] for sample in self.samples]
        unique, counts = np.unique(labels, return_counts=True)

        return {
            "data_dir": str(self.data_dir),
            "num_files": len(self.files),
            "num_samples": len(self.samples),
            "subjects": sorted({sample["subject"] for sample in self.samples}),
            "runs": sorted({sample["run"] for sample in self.samples}),
            "channel_set": self.channel_set,
            "num_channels": self.num_channels,
            "channel_names": self.channel_names,
            "sfreq": self.sfreq,
            "target_sfreq": self.target_sfreq,
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "num_timepoints": self.num_timepoints,
            "label_counts": dict(zip(unique.tolist(), counts.tolist())),
        }