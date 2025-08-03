# Beat-Synced Squares (Offline)

Turn any video – or your webcam feed – into a pulsing network of labelled squares that dance to the beat.

## Features
• Beat-driven spawning of squares with ORB keypoints  
• LK optical-flow tracking with subtle jitter for organic motion  
• Optional ambient "noise" spawns so visuals never fall silent  
• Neighbor-link edges for a living graph aesthetic  
• Per-square color-inversion, random alphanumeric labels, vertical text option  

## Installation
Requirements: Python 3.9+, FFmpeg on PATH.

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

### 1. Offline processing
```bash
python main.py \
  --input sample_data/playing_dead.mp4 \
  --output output/playing_dead_boxes.mp4 \
  --life-frames 10 \
  --pts-per-beat 20 \
  --ambient-rate 5.0 \
  --jitter-px 0.5 \
  --neighbor-links 3
```

## How it works
1. Extract audio, detect onsets with Librosa.  
2. At each onset, ORB keypoints are sampled; a subset spawns squares.  
3. Squares are tracked across frames with Lucas-Kanade optical flow; small Gaussian jitter adds life.  
4. Edges connect each square to its nearest neighbors.  
5. Squares invert colors within their bounds, display a random label, and expire after `life_frames`.  
6. Ambient Poisson spawns keep visuals active during silence.

## Project layout
```
Effect_v1
├── main.py          # offline renderer
├── sample_data/     # demo video
├── output/          # rendered results go here
├── requirements.txt
```
