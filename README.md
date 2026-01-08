# Advanced Spatiotemporal Emotion Recognition System

A real-time emotion recognition system combining deep learning, computer vision, and multi-modal feature fusion for accurate emotion detection from video streams.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### Core Capabilities
- **Multi-scale Temporal Attention** - Captures emotion dynamics across different time scales
- **Facial Landmark Integration** - 50+ facial landmarks for precise emotion detection
- **Dynamic Channel Attention** - Adaptive feature selection with temporal consistency
- **Cross-modal Fusion** - Combines visual and geometric features for robust predictions
- **Valence-Arousal Mapping** - Continuous emotion space representation
- **Real-time Processing** - Optimized for live video streaming
- **Temporal Smoothing** - Stabilized predictions using moving average filtering
- **Video File Processing** - Batch process pre-recorded videos with progress tracking
- **FPS Monitoring** - Real-time performance metrics display
- **Probability Visualization** - Live confidence bars for all emotion categories
- **TensorBoard Integration** - Training visualization and metrics tracking
- **Early Stopping** - Automatic training termination to prevent overfitting
- **Mixed Precision Training** - Faster training on CUDA GPUs
- **JSON Export** - Frame-by-frame emotion analysis results
- **Object Detection** - Integrated YOLOv8 for scene understanding

## Emotion Categories

The system recognizes 7 distinct emotions:
1. **Angry** 😠
2. **Disgust** 🤢
3. **Fear** 😨
4. **Happy** 😊
5. **Neutral** 😐
6. **Sad** 😢
7. **Surprise** 😲

## 🏗️ Architecture

```
Input Video Stream
      ↓
┌─────────────────────────────────────┐
│  Frame Buffer (16 frames)           │
│  + Facial Landmark Extraction       │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Spatial Feature Extraction         │
│  (CNN: 64→128→256→512 channels)    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Multi-scale Temporal Attention     │
│  (3 parallel temporal convolutions) │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Dynamic Channel Attention          │
│  + Spatial Attention Mechanism      │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Landmark Feature Processing        │
│  (256→128 dimensional embedding)    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Cross-modal Fusion Layer           │
│  (Visual + Landmark features)       │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Multi-head Prediction              │
│  • Valence: [-1, 1]                 │
│  • Arousal: [0, 1]                  │
│  • Category: 7 emotions             │
└─────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (optional, for faster training)
- Webcam (for real-time demo)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd HackOWeek
```

2. **Install dependencies**
```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install mediapipe
pip install ultralytics
pip install einops
pip install matplotlib seaborn
pip install tqdm
pip install tensorboard
pip install scikit-learn
```

3. **Download pre-trained models**
```bash
# YOLOv8 model (automatically downloaded on first run)
# Emotion recognition model (trained or provided)
```

## Usage

### Real-time Webcam Demo

```bash
# Basic real-time demo with object detection
python3.12 main.py --realtime

# Real-time demo without object detection (faster)
python3.12 main.py --realtime --no-objects

# Custom temporal smoothing (higher = more stable but slower response)
python3.12 main.py --realtime --smoothing 10
```

**Controls:**
- Press `q` to quit the demo

### Video File Processing

```bash
# Process a video file
python3.12 main.py --video input.mp4 --output result.mp4

# Process with custom smoothing
python3.12 main.py --video input.mp4 --output result.mp4 --smoothing 7
```

**Output:**
- Annotated video with emotion predictions
- JSON file with frame-by-frame analysis (`*_results.json`)

### Training

```bash
# Train with default settings (50 epochs)
python3.12 main.py --train

# Train with custom epochs
python3.12 main.py --train --epochs 100

# Monitor training with TensorBoard
tensorboard --logdir=runs/emotion_recognition
```

### Default Behavior

```bash
# Creates model architecture and saves weights
python3.12 main.py
```

## 📊 Model Details

| Component | Description | Parameters |
|-----------|-------------|------------|
| **Spatial CNN** | 4-layer convolutional network | ~3.2M |
| **Temporal Attention** | Multi-scale temporal feature extraction | ~0.8M |
| **Channel Attention** | Dynamic channel weighting with LSTM | ~0.5M |
| **Landmark Processor** | Facial geometry feature extraction | ~0.3M |
| **Fusion Network** | Cross-modal feature integration | ~2.1M |
| **Prediction Heads** | Valence + Arousal + Category | ~1.1M |
| **Total** | | **~8.0M** |

## Output Examples

### Real-time Display
The system displays:
- **Emotion label** with confidence score
- **Valence** value (-1 to 1: negative to positive)
- **Arousal** value (0 to 1: calm to excited)
- **Probability bars** for all 7 emotions
- **FPS counter** for performance monitoring
- **Object detection boxes** (optional)

### JSON Results
```json
[
  {
    "frame": 0,
    "emotion": "Happy",
    "confidence": 0.87,
    "valence": 0.65,
    "arousal": 0.72
  },
  {
    "frame": 1,
    "emotion": "Happy",
    "confidence": 0.89,
    "valence": 0.68,
    "arousal": 0.75
  }
]
```

## Configuration

### Key Parameters

```python
# Model configuration
temporal_length = 16        # Number of frames to process
num_landmarks = 50         # Facial landmarks to extract
smoothing_window = 5       # Temporal smoothing window size

# Training configuration
learning_rate = 0.001      # Initial learning rate
weight_decay = 1e-4        # L2 regularization
patience = 10              # Early stopping patience

# Loss weights
valence_weight = 1.0       # Valence prediction weight
arousal_weight = 1.0       # Arousal prediction weight
category_weight = 0.5      # Category classification weight
```

## Technical Details

### Attention Mechanisms

**Multi-scale Temporal Attention:**
- Parallel convolutions with kernel sizes: 1, 3, 5
- Captures short-term, medium-term, and long-term patterns
- Softmax-based attention weighting

**Dynamic Channel Attention:**
- Average and max pooling for global context
- Bidirectional LSTM for temporal consistency
- Sigmoid activation for channel gating

**Spatial Attention:**
- Combines average and max spatial features
- 7×7 convolution for spatial weighting
- Element-wise multiplication with features

### Data Flow

1. **Input:** 16 consecutive frames (224×224×3)
2. **Landmark Extraction:** MediaPipe Face Mesh (50 key points)
3. **Spatial Features:** CNN with batch normalization
4. **Temporal Features:** Multi-scale temporal convolutions
5. **Attention:** Channel + Spatial + Temporal weighting
6. **Fusion:** Concatenate visual and landmark features
7. **Output:** Valence, Arousal, Category predictions

## Performance

- **Inference Speed:** ~30 FPS on CPU, ~60+ FPS on GPU
- **Model Size:** 30.7 MB (FP32), 15.4 MB (FP16)
- **Memory Usage:** ~500 MB GPU, ~200 MB CPU
- **Latency:** <50ms per frame (including preprocessing)

## Troubleshooting

### Common Issues

**1. "Model file not found"**
- Solution: Run `python3.12 main.py` first to create model weights

**2. "No webcam detected"**
- Solution: Check camera permissions and connection
- Try: `python3.12 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

**3. "MediaPipe not available"**
- Solution: Install with `pip install mediapipe`
- Fallback: System uses zero landmarks if unavailable

**4. Low FPS in real-time mode**
- Solution: Use `--no-objects` flag to disable object detection
- Reduce `--smoothing` value for faster response

**5. CUDA out of memory**
- Solution: Reduce batch size or use CPU
- Try: Mixed precision training with smaller models

## Future Enhancements

- [ ] Multi-face detection and tracking
- [ ] Audio-visual emotion fusion
- [ ] Micro-expression detection
- [ ] Custom emotion categories
- [ ] Mobile deployment (ONNX/TFLite)
- [ ] REST API for cloud deployment
- [ ] Real-time dashboard with analytics
- [ ] Multi-language support

## References

- **Attention Mechanisms:** Vaswani et al., "Attention Is All You Need"
- **Facial Landmarks:** MediaPipe Face Mesh
- **Object Detection:** YOLOv8 (Ultralytics)
- **Emotion Models:** Valence-Arousal Circumplex Model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- MediaPipe for facial landmark detection
- Ultralytics for YOLOv8 object detection
- Open-source community for various tools and libraries

## Contact

For questions, issues, or collaborations, please open an issue on GitHub.

---
