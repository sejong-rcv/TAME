import torch as th
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from decord import VideoReader, cpu, gpu

class RawVideoExtractorDecord():
    def __init__(self, centercrop=False, size=224, framerate=-1, use_gpu=False, gpu_id=0):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _build_indices(self, nframes, fps, sample_fp, start_time, end_time):
        # time -> frame range
        if start_time is None:
            start_frame = 0
        else:
            start_frame = max(0, int(start_time * fps))

        if end_time is None:
            end_frame = nframes - 1
        else:
            end_frame = min(nframes - 1, int(end_time * fps))

        if end_frame < start_frame:
            return []

        # sample_fp: "frames per second" like your original code
        if sample_fp <= 0:
            # fallback: take ~fps frames per second (dense)
            sample_fp = int(round(fps)) if fps > 0 else 1

        fps_i = max(1, int(round(fps)))  # approximate
        interval = max(1, fps_i // sample_fp)
        inds_in_sec = np.arange(0, fps_i, interval)[:sample_fp]

        start_sec = start_frame // fps_i
        end_sec = end_frame // fps_i

        indices = []
        for sec in range(start_sec, end_sec + 1):
            base = sec * fps_i
            for off in inds_in_sec:
                idx = base + int(off)
                if start_frame <= idx <= end_frame and idx < nframes:
                    indices.append(idx)

        # unique & sorted (avoid duplicates near boundaries)
        indices = sorted(set(indices))
        return indices

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        ctx = gpu(self.gpu_id) if self.use_gpu else cpu(0)
        vr = VideoReader(video_file, ctx=ctx)

        nframes = len(vr)
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
        if fps <= 1e-6:
            fps = 30.0

        indices = self._build_indices(nframes, fps, sample_fp, start_time, end_time)
        if len(indices) == 0:
            return {'video': th.zeros(1)}

        # batch decode (fast)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3), RGB
        images = [preprocess(Image.fromarray(frames[i])) for i in range(frames.shape[0])]
        video_data = th.tensor(np.stack(images))
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        return self.video_to_tensor(video_path, self.transform,
                                    sample_fp=self.framerate,
                                    start_time=start_time, end_time=end_time)

    def process_raw_data(self, raw_video_data):
        ts = raw_video_data.size()
        return raw_video_data.view(-1, 1, ts[-3], ts[-2], ts[-1])

    def process_frame_order(self, raw_video_data, frame_order=0):
        if frame_order == 1:
            idx = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[idx, ...]
        elif frame_order == 2:
            idx = np.arange(raw_video_data.size(0))
            np.random.shuffle(idx)
            raw_video_data = raw_video_data[idx, ...]
        return raw_video_data

RawVideoExtractor = RawVideoExtractorDecord



