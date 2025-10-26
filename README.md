# Audio Upscaler

Audio Upscaler is a U-Net-style convolutional autoencoder trained to "upscale" audio using AI. For example, it can convert from an 8-bit 11KHz IFF file to a 16-bit 44.1KHz WAV file. The alpha weights included here are a placeholder and are mostly useless at the moment—this is a running project, training on approximately 200,000 audio files. The weights will eventually be useful for improving sample rates and depth as well as fixing issues associated with audio quality for old and obscure low-quality audio formats.

## Project Status

**Alpha 0.001** - Early development stage

- Alpha weights included are mostly non-functional placeholders
- Training ongoing on ~200,000 audio files
- Weights will improve significantly in future releases
- Not recommended for production use yet

## Description

Audio Upscaler uses deep learning to enhance audio quality by increasing sample rate and bit depth. The U-Net architecture allows the model to learn features at multiple scales, making it effective for audio restoration and quality improvement tasks.

### Use Cases

- Convert old 8-bit, low sample rate audio to modern formats
- Enhance audio from obsolete formats (IFF, SoundBlaster, etc.)
- Improve quality of digitized cassettes, vinyl records, and old game audio
- Restore archived low-quality audio files

## Architecture

The model uses a **U-Net-style convolutional autoencoder** consisting of:
- Encoder: Downsampling path to capture multi-scale features
- Bottleneck: Central feature representation
- Decoder: Upsampling path for reconstruction
- Skip connections: Preserve fine details from encoder to decoder

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- SciPy
- librosa (for audio processing)

## Installation

```bash
git clone https://github.com/yourusername/AudioUpscaler.git
cd AudioUpscaler
pip install -r requirements.txt
```

Or install dependencies manually:
```bash
pip install torch numpy scipy librosa
```

## Usage

### Loading the Model

```python
import torch
from audio_upscaler import AudioUpscaler

# Load the pre-trained model
model = AudioUpscaler()
model.load_state_dict(torch.load('AudioUpscaler-alpha0.001.pt'))
model.eval()
```

### Upscaling Audio

```python
import librosa
import numpy as np

# Load audio
audio, sr = librosa.load('input.wav', sr=11025, mono=True)

# Prepare input (normalize to [-1, 1])
audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)

# Upscale
with torch.no_grad():
    upscaled = model(audio_tensor)

# Convert back to numpy and save
output = upscaled.squeeze().numpy()
librosa.output.write_wav('output.wav', output, sr=44100)
```

## Model Files

### AudioUpscaler-alpha0.001.pt

- **Size**: ~181 MB
- **Format**: PyTorch state dictionary
- **Status**: Alpha weights (placeholder)
- **Recommended**: For testing and development only

## Training Data

The model is being trained on approximately **200,000 audio files** covering:
- Various sample rates (8kHz, 11.025kHz, 16kHz, 22.05kHz, 44.1kHz)
- Multiple bit depths (8-bit, 16-bit)
- Diverse audio formats (IFF, WAV, MP3, FLAC, etc.)
- Wide range of audio content (speech, music, effects, etc.)

## Future Improvements

- [ ] Expand training dataset
- [ ] Improve model architecture
- [ ] Add multi-channel support
- [ ] Optimize inference speed
- [ ] Release beta weights (v0.5)
- [ ] Release production weights (v1.0)
- [ ] Support for real-time upscaling
- [ ] CLI tool for batch processing

## Limitations

- **Alpha weights are mostly non-functional** - do not expect good results with current weights
- Current model: single-channel audio only
- Fixed input/output sample rates (work in progress)
- GPU recommended for faster processing
- Long audio files may require memory optimization

## Contributing

This is an active research and development project. We welcome:
- Feedback on audio quality improvements
- Bug reports and feature requests
- Contributions to training data collection
- Performance optimization suggestions

## References

U-Net architecture papers and implementations:
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation
- Various audio upsampling and enhancement techniques

## License

[Specify your license - e.g., MIT, GPL, Apache 2.0]

## Roadmap

**v0.001 (Current - Alpha)**
- Initial model and weights
- Basic U-Net architecture

**v0.5 (Beta - Planned)**
- Improved training on full dataset
- Better generalization across formats
- Performance optimizations

**v1.0 (Production - Planned)**
- Production-grade weights
- Full feature support
- Comprehensive documentation
- Optimization and deployment guides

---

*A deep learning project for audio quality enhancement and restoration. Stay tuned for improved weights as training progresses.*
