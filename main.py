
# from __future__ import annotations must be first
from __future__ import annotations

# Standard library imports
import time
import argparse
import logging
import os
import random
import tempfile
from pathlib import Path
import uuid

# Third-party imports
import cv2
import librosa
import moviepy.editor as mpy
import numpy as np
# ---------------------------------------------------------------------------
# Real-time webcam effect
# ---------------------------------------------------------------------------

def run_realtime_effect(
    *,
    fps: float = 30.0,
    pts_per_beat: int = 10,
    ambient_rate: float = 5.0,
    jitter_px: float = 0.5,
    life_frames: int = 10,
    min_size: int = 15,
    max_size: int = 40,
    neighbor_links: int = 3,
    orb_fast_threshold: int = 20,
    bell_width: float = 4.0,
    seed: int | None = None,
    bpm: float = 120.0,
    camera_index: int = 0,
):
    """Run the effect in real time using a webcam and a simple beat timer."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)
    active: list[TrackedPoint] = []
    prev_gray: np.ndarray | None = None
    last_beat_time = time.time()
    beat_interval = 60.0 / bpm

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror for webcam
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Track existing points with LK optical flow
        if prev_gray is not None and active:
            prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=3
            )
            new_active: list[TrackedPoint] = []
            for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue
                x, y = new_pt
                if 0 <= x < w and 0 <= y < h and tp.life > 0:
                    tp.pos = new_pt
                    tp.life -= 1
                    if jitter_px > 0:
                        tp.pos += np.random.normal(0, jitter_px, size=2)
                        tp.pos[0] = np.clip(tp.pos[0], 0, w - 1)
                        tp.pos[1] = np.clip(tp.pos[1], 0, h - 1)
                    new_active.append(tp)
            active = new_active

        # 2. Spawn new points on each beat (simple timer)
        now = time.time()
        if now - last_beat_time >= beat_interval:
            kps = orb.detect(gray, None)
            kps = sorted(kps, key=lambda k: k.response, reverse=True)
            target_spawn = random.randint(1, pts_per_beat)
            spawned = 0
            for kp in kps:
                if spawned >= target_spawn:
                    break
                x, y = kp.pt
                if any(np.linalg.norm(tp.pos - (x, y)) < 10 for tp in active):
                    continue
                size = _sample_size_bell(min_size, max_size, bell_width)
                r = random.random()
                if r < 0.33:
                    label = ''.join(random.choices('ABCDEF0123456789', k=6))
                elif r < 0.66:
                    label = str(random.randint(1, 999))
                else:
                    label = str(uuid.uuid4())[:8]
                font_scale = random.uniform(1.0, 1.8)
                text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])
                vertical = random.random() < 0.25
                active.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))
                spawned += 1
            last_beat_time = now

        # 2b. Ambient random spawns each frame (noise)
        if ambient_rate > 0:
            noise_n = np.random.poisson(ambient_rate / fps)
            for _ in range(noise_n):
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                size = _sample_size_bell(min_size, max_size, bell_width)
                label_choices = [
                    ''.join(random.choices('ABCDEF0123456789', k=6)),
                    str(random.randint(1, 999)),
                    str(uuid.uuid4())[:8],
                ]
                label = random.choice(label_choices)
                font_scale = random.uniform(1.0, 1.8)
                text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])
                vertical = random.random() < 0.25
                active.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))

        # 3. Draw edges (nearest neighbors)
        coords = [tp.pos for tp in active]
        for i, p in enumerate(coords):
            dists = [ (j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i ]
            dists.sort(key=lambda x: x[1])
            for j, _ in dists[:neighbor_links]:
                cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (255, 255, 255), 1)

        # 4. Draw squares for each point
        for tp in active:
            x, y = tp.pos
            s = tp.size
            tl = (int(x - s // 2), int(y - s // 2))
            br = (int(x + s // 2), int(y + s // 2))
            tl_clamped = (max(0, tl[0]), max(0, tl[1]))
            br_clamped = (min(w, br[0]), min(h, br[1]))
            roi = frame[tl_clamped[1]:br_clamped[1], tl_clamped[0]:br_clamped[0]]
            if roi.size:
                frame[tl_clamped[1]:br_clamped[1], tl_clamped[0]:br_clamped[0]] = 255 - roi
            cv2.rectangle(frame, tl, br, (255, 255, 255), 1)
            if tp.vertical:
                y_cursor = tl[1] + 2
                line_height = int(12 * tp.font_scale)
                for ch in tp.label:
                    if y_cursor > br[1] - 2:
                        break
                    cv2.putText(
                        frame,
                        ch,
                        (tl[0] + 2, y_cursor),
                        cv2.FONT_HERSHEY_PLAIN,
                        tp.font_scale,
                        tp.text_color,
                        1,
                        cv2.LINE_AA,
                    )
                    y_cursor += line_height
            else:
                cv2.putText(
                    frame,
                    tp.label,
                    (tl[0] + 2, br[1] - 4),
                    cv2.FONT_HERSHEY_PLAIN,
                    tp.font_scale,
                    tp.text_color,
                    1,
                    cv2.LINE_AA,
                )

        prev_gray = gray
        cv2.imshow('Real-Time Effect', frame)
        key = cv2.waitKey(int(1000 // fps)) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ...existing code...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_audio(video_path: Path, sr: int = 22050) -> Path:
    """Write the audio track of *video_path* to a temporary wav file and return its path."""
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    # MoviePy uses FFmpeg under the hood
    clip = mpy.VideoFileClip(str(video_path))
    clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None, verbose=False)
    return wav_path


def _detect_onsets(wav_path: Path, sr: int = 22050) -> np.ndarray:
    """Return an array of onset times (in seconds) detected in the audio file."""
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")


class Square:
    """Data container for an on-screen square."""

    def __init__(self, born_at: float, x: int, y: int, size: int, idx: int):
        self.born_at = born_at
        self.x = x
        self.y = y
        self.size = size
        self.idx = idx

    def age(self, now: float) -> float:
        return now - self.born_at


# ---------------------------------------------------------------------------
# Tracked points pipeline
# ---------------------------------------------------------------------------

class TrackedPoint:
    """A feature point tracked across successive frames."""

    def __init__(
        self,
        pos: tuple[float, float],
        life: int,
        size: int,
        label: str,
        font_scale: float,
        text_color: tuple[int, int, int],
        vertical: bool,
    ):
        self.pos = np.array(pos, dtype=np.float32)  # shape (2,)
        self.life = life  # remaining frames
        self.size = size  # constant box size
        self.label = label  # text label shown inside box
        self.font_scale = font_scale  # scale factor for cv2.putText
        self.text_color = text_color  # BGR color tuple
        self.vertical = vertical  # whether to render text vertically


def render_tracked_effect(
    video_in: Path,
    video_out: Path,
    audio_in: Path | None = None,
    *,
    fps: float | None,
    pts_per_beat: int,
    ambient_rate: float,
    jitter_px: float,
    life_frames: int,
    min_size: int,
    max_size: int,
    neighbor_links: int,
    orb_fast_threshold: int,
    bell_width: float,
    seed: int | None,
):
    """Generate a video where feature points are spawned on beats and tracked with LK optical flow."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clip = mpy.VideoFileClip(str(video_in))
    if fps is None:
        fps = clip.fps


    # Use custom audio for beat detection if provided
    if audio_in is not None:
        # Convert to wav for librosa if needed
        if str(audio_in).lower().endswith('.wav'):
            wav_path = audio_in
        else:
            tmp_dir = Path(tempfile.mkdtemp())
            wav_path = tmp_dir / "temp_audio.wav"
            audioclip = mpy.AudioFileClip(str(audio_in))
            audioclip.write_audiofile(str(wav_path), fps=22050, logger=None, verbose=False)
    else:
        wav_path = _extract_audio(video_in)
    onset_times = _detect_onsets(wav_path)
    logging.info("%d onsets detected", len(onset_times))

    # ORB detector for spawning
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)

    active: list[TrackedPoint] = []
    onset_idx = 0
    prev_gray: np.ndarray | None = None

    def make_frame(t: float):
        nonlocal prev_gray, onset_idx, active
        frame = clip.get_frame(t).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        f_idx = int(round(t * fps))

        # 1. Track existing points with LK optical flow
        if prev_gray is not None and active:
            prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=3
            )
            new_active: list[TrackedPoint] = []
            for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue
                x, y = new_pt
                if 0 <= x < w and 0 <= y < h and tp.life > 0:
                    tp.pos = new_pt
                    tp.life -= 1
                    if jitter_px > 0:
                        tp.pos += np.random.normal(0, jitter_px, size=2)
                        tp.pos[0] = np.clip(tp.pos[0], 0, w - 1)
                        tp.pos[1] = np.clip(tp.pos[1], 0, h - 1)
                    new_active.append(tp)
            active = new_active

        # 2. Spawn new points on each beat that has passed
        while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
            logging.debug("Beat @ %.2fs — spawning points", onset_times[onset_idx])
            kps = orb.detect(gray, None)
            # Sort by response strength
            kps = sorted(kps, key=lambda k: k.response, reverse=True)
            target_spawn = random.randint(1, pts_per_beat)
            spawned = 0
            for kp in kps:
                if spawned >= target_spawn:
                    break
                x, y = kp.pt
                if any(np.linalg.norm(tp.pos - (x, y)) < 10 for tp in active):
                    continue  # too close to an existing point
                size = _sample_size_bell(min_size, max_size, bell_width)
                # generate random label
                r = random.random()
                if r < 0.33:
                    label = ''.join(random.choices('ABCDEF0123456789', k=6))
                elif r < 0.66:
                    label = str(random.randint(1, 999))
                else:
                    label = str(uuid.uuid4())[:8]

                # text styling attributes
                font_scale = random.uniform(1.0, 1.8)  # larger text than before
                text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])  # white/black/purple (BGR)
                vertical = random.random() < 0.25  # 25% chance to render vertically

                active.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))
                spawned += 1
            logging.info("Spawned %d points", spawned)
            onset_idx += 1

        # 2b. Ambient random spawns each frame (noise)
        if ambient_rate > 0:
            noise_n = np.random.poisson(ambient_rate / fps)
            for _ in range(noise_n):
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                size = _sample_size_bell(min_size, max_size, bell_width)
                label_choices = [
                    ''.join(random.choices('ABCDEF0123456789', k=6)),
                    str(random.randint(1, 999)),
                    str(uuid.uuid4())[:8],
                ]
                label = random.choice(label_choices)

                # text styling attributes (ambient)
                font_scale = random.uniform(1.0, 1.8)
                text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])
                vertical = random.random() < 0.25

                active.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))

        # 3. Draw edges (nearest neighbors)
        coords = [tp.pos for tp in active]
        for i, p in enumerate(coords):
            dists = [ (j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i ]
            dists.sort(key=lambda x: x[1])
            for j, _ in dists[:neighbor_links]:
                cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (255, 255, 255), 1)

        # 4. Draw squares for each point
        for tp in active:
            x, y = tp.pos
            s = tp.size
            tl = (int(x - s // 2), int(y - s // 2))
            br = (int(x + s // 2), int(y + s // 2))
            # Clamp coordinates to frame boundaries
            tl_clamped = (max(0, tl[0]), max(0, tl[1]))
            br_clamped = (min(w, br[0]), min(h, br[1]))
            # Invert colors inside box for pop effect
            roi = frame[tl_clamped[1]:br_clamped[1], tl_clamped[0]:br_clamped[0]]
            if roi.size:
                frame[tl_clamped[1]:br_clamped[1], tl_clamped[0]:br_clamped[0]] = 255 - roi
            cv2.rectangle(frame, tl, br, (255, 255, 255), 1)
            # draw label text with new styling
            if tp.vertical:
                y_cursor = tl[1] + 2
                line_height = int(12 * tp.font_scale)
                for ch in tp.label:
                    if y_cursor > br[1] - 2:
                        break  # stop if text would overflow box
                    cv2.putText(
                        frame,
                        ch,
                        (tl[0] + 2, y_cursor),
                        cv2.FONT_HERSHEY_PLAIN,
                        tp.font_scale,
                        tp.text_color,
                        1,
                        cv2.LINE_AA,
                    )
                    y_cursor += line_height
            else:
                cv2.putText(
                    frame,
                    tp.label,
                    (tl[0] + 2, br[1] - 4),
                    cv2.FONT_HERSHEY_PLAIN,
                    tp.font_scale,
                    tp.text_color,
                    1,
                    cv2.LINE_AA,
                )

        prev_gray = gray
        return frame

    out_clip = mpy.VideoClip(make_frame, duration=clip.duration)
    # Use custom audio for output if provided
    if audio_in is not None:
        audioclip = mpy.AudioFileClip(str(audio_in)).set_duration(clip.duration)
        out_clip = out_clip.set_audio(audioclip)
    else:
        out_clip = out_clip.set_audio(clip.audio)
    out_clip.write_videofile(str(video_out), fps=fps, codec="libx264", audio_codec="aac")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sample_size_bell(min_s: int, max_s: int, width_div: float = 6.0) -> int:
    """Sample size with adjustable bell width; smaller width_div => wider distribution."""
    mean = (min_s + max_s) / 2.0
    sigma = (max_s - min_s) / width_div
    for _ in range(10):
        val = np.random.normal(mean, sigma)
        if min_s <= val <= max_s:
            return int(val)
    # fallback clamp
    return int(np.clip(val, min_s, max_s))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Overlay beat-synced boxes + lines on a video.")
    p.add_argument("-i", "--input", type=Path, default=Path("sample_data/playing_dead.mp4"), help="Input video file")
    p.add_argument("-o", "--output", type=Path, default=Path("output_with_boxes.mp4"), help="Output video file")
    p.add_argument("--audio", type=Path, default=None, help="Optional audio file for beat detection and output")
    p.add_argument("--fps", type=float, default=None, help="FPS for output (default: same as source)")
    p.add_argument("--life-frames", type=int, default=10, help="How many frames a point remains alive (short for crazy mode)")
    p.add_argument("--pts-per-beat", type=int, default=20, help="Maximum new points to spawn on each beat (actual number is random up to this)")
    p.add_argument("--ambient-rate", type=float, default=5.0, help="Average number of random points spawned per second even in silence")
    p.add_argument("--jitter-px", type=float, default=0.5, help="Per-frame positional jitter in pixels for organic motion")
    p.add_argument("--min-size", type=int, default=15, help="Minimum square size in pixels")
    p.add_argument("--max-size", type=int, default=40, help="Maximum square size in pixels")
    p.add_argument("--neighbor-links", type=int, default=3, help="Number of neighbor edges per point")
    p.add_argument("--orb-fast-threshold", type=int, default=20, help="FAST threshold for ORB detector (lower = more keypoints)")
    p.add_argument("--bell-width", type=float, default=4.0, help="Divisor controlling bell curve width (smaller -> wider)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    p.add_argument('--realtime', action='store_true', help='Run in real-time webcam mode')
    p.add_argument('--bpm', type=float, default=120.0, help='Beats per minute for real-time mode')
    p.add_argument('--camera-index', type=int, default=0, help='Webcam index for real-time mode')
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(message)s")

    if getattr(args, 'realtime', False):
        run_realtime_effect(
            fps=args.fps or 30.0,
            pts_per_beat=args.pts_per_beat,
            ambient_rate=args.ambient_rate,
            jitter_px=args.jitter_px,
            life_frames=args.life_frames,
            min_size=args.min_size,
            max_size=args.max_size,
            neighbor_links=args.neighbor_links,
            orb_fast_threshold=args.orb_fast_threshold,
            bell_width=args.bell_width,
            seed=args.seed,
            bpm=args.bpm,
            camera_index=args.camera_index,
        )
    else:
        render_tracked_effect(
            video_in=args.input,
            video_out=args.output,
            audio_in=args.audio,
            fps=args.fps,
            pts_per_beat=args.pts_per_beat,
            ambient_rate=args.ambient_rate,
            jitter_px=args.jitter_px,
            life_frames=args.life_frames,
            min_size=args.min_size,
            max_size=args.max_size,
            neighbor_links=args.neighbor_links,
            orb_fast_threshold=args.orb_fast_threshold,
            bell_width=args.bell_width,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
