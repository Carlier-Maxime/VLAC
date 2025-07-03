import json
import os
import argparse
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm
import cv2

from vlac.dataset.format.format_dict import FormatDictDataset


class FormatMinerlDataset(FormatDictDataset):
    def __init__(self, **_):
        super().__init__(**_)
        self.base_path = None

    def get_iterator(self, input_path: str, parquet_size: int) -> Iterable:
        self.base_path = input_path
        names = []
        with os.scandir(input_path) as it:
            for entry in tqdm(it, desc='scan minerl dataset', unit='entry'):
                if not entry.is_file():
                    pass
                ext = os.path.splitext(entry.name)
                basename = os.path.basename(ext[0])
                if ext != ".mp4":
                    pass
                names.append(basename)
        return tqdm(names, desc='format minerl dataset', unit='video')

    @staticmethod
    def get_nb_frames_manual(cap: cv2.VideoCapture):
        nb_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            nb_frames += 1
        return nb_frames

    @staticmethod
    def read_jsonl(path) -> List[dict]:
        lines = []
        with open(path, 'r') as f:
            for line in f.readlines():
                if line is None: break
                if line.strip() == "": continue
                lines.append(json.loads(line))
        return lines

    def make_df(self, data, step_data: argparse.Namespace) -> pd.DataFrame | None:
        vid_path = f'{self.base_path}/{data}.mp4'
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"WARNING: {data}.mp4 not readable, skipping it.")
            return None
        jsonl_path = f'{self.base_path}/{data}.jsonl'
        if not os.path.exists(jsonl_path):
            print(f"WARNING: {data}.jsonl not found, skipping it.")
            return None
        infos = self.read_jsonl(jsonl_path)
        vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if vid_len != len(infos)+1: vid_len = self.get_nb_frames_manual(cap)
        assert vid_len == len(infos)+1, f"video length mismatch: {vid_len} != {len(infos)+1}"
        cap.release()
        with open(vid_path, "rb") as f:
            video_bytes = f.read()
        with open(jsonl_path, "rb") as f:
            jsonl_bytes = f.read()
        return super().make_df({
            'video.mp4': video_bytes,
            'vid_len': vid_len,
            'fps': fps,
            'vid_width': vid_width,
            'vid_height': vid_height,
            'task_desc': None,
            'infos.jsonl': jsonl_bytes,
        }, step_data)

    def resume_to_samples(self, iterator: Iterable, resume_samples: int) -> Iterable:
        if isinstance(iterator, tqdm):
            base_iter = iterator.iterable
            desc = iterator.desc
            unit = iterator.unit
        else:
            base_iter = iterator
            desc = None
            unit = 'it'
        return tqdm(base_iter[resume_samples:], desc=desc, unit=unit)

    def get_infos_of_parquet(self, df: pd.DataFrame) -> dict:
        infos = super().get_infos_of_parquet(df)
        infos['frames'] = int(df['vid_len'].sum())
        return infos


if __name__ == "__main__":
    FormatMinerlDataset.format_from_args()
