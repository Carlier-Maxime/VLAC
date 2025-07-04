import io
import os
import tempfile
from typing import Union

import ffmpeg
import numpy as np
import torch


class VideoReader:
    def __init__(self, video_bytes: Union[bytes, io.BytesIO], width: int = None, height: int = None):
        self.video_bytes = video_bytes if isinstance(video_bytes, bytes) else video_bytes.read()
        self.__to_memfd()

        if width is None or height is None:
            width, height = self._get_video_resolution()

        self.width = width
        self.height = height
        self.frame_size = self.width * self.height * 3

        self.process = (
            ffmpeg
            .input(self.memfd_path, probesize='100M', analyzeduration='100M')
            .output('pipe:1', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdin=False, pipe_stdout=True, pipe_stderr=True)
        )

    def __to_memfd(self):
        try:
            self.fd = os.memfd_create("video", flags=os.MFD_CLOEXEC)
            os.write(self.fd, self.video_bytes)
            os.lseek(self.fd, 0, os.SEEK_SET)
            self.memfd_path = f"/proc/self/fd/{self.fd}"
            self._using_memfd = True
            return
        except AttributeError or OSError:
            pass

        try:
            self.__to_temp_file(directory="/dev/shm")
            return
        except (FileNotFoundError, PermissionError):
            pass

        self.__to_temp_file()

    def __to_temp_file(self, directory: str = None):
        tmpfile = tempfile.NamedTemporaryFile(prefix="video_", dir=directory, delete=False)
        tmpfile.write(self.video_bytes)
        tmpfile.flush()
        tmpfile.seek(0)
        self.memfd_path = tmpfile.name
        self._using_memfd = False
        tmpfile.close()

    def _get_video_resolution(self):
        probe = ffmpeg.probe(self.memfd_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return int(video_stream['width']), int(video_stream['height'])

    def __iter__(self):
        return self

    def __next__(self):
        in_bytes = self.process.stdout.read(self.frame_size)
        if not in_bytes:
            self.close()
            raise StopIteration
        frame = np.frombuffer(in_bytes, np.uint8).copy().reshape((self.height, self.width, 3))
        return torch.from_numpy(frame).permute(2, 0, 1)

    def close(self):
        if self.process:
            self.process.stdout.close()
            self.process.stderr.close()
            self.process.wait()
            self.process = None
        if self.fd:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        self.close()
