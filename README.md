# TNMT Audio Processor

TNMT Audio Processor is an implementation of the audio processing pipeline described in the article "Clinical trial on tonal tinnitus with tailor-made notched music training" by Pantev et al. (2016). The pipeline includes a multi-band auto equalizer, a notch filter around tinnitus frequency, and edge amplification at the notch filter boundaries.
Research paper reference: https://link.springer.com/content/pdf/10.1186/s12883-016-0558-7.pdf

## Prerequisites

- Python 3.6 or higher
- Install required packages with:
  ```sh
  pip install -r requirements.txt
- Install BlackHole virtual device for audio loopback (macOS only):
  - Download BlackHole from https://existential.audio/blackhole/
  - Open the downloaded .pkg file and follow the installation instructions.
  - Open the "Audio MIDI Setup" application on your Mac
  - Locate "BlackHole 2ch" device and set it as the default output device
    
## Usage

- Run the script tnmt.py
- Select BlackHole as the input device and your soundcard as the output device
- Listen to the music!
- Some latency is required to keep up the quality of the processed audio... so it's not recomended to use this program for watching videos...

## What if I don't suffer from tinnitus?

- This program seems to work well as an audio enhancer too! Try it on your favourite music and hear the difference. In this case you might want to disable the notch filtering via the related checkbox in the GUI.
