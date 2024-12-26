# TNMT Audio Processor

- TNMT Audio Processor is an implementation of the audio processing pipeline described in the article "Clinical trial on tonal tinnitus with tailor-made notched music training" by Pantev et al. (2016). The pipeline includes a multi-band auto equalizer, a notch filter around tinnitus frequency, and edge amplification at the notch filter boundaries.
Research paper reference: https://link.springer.com/content/pdf/10.1186/s12883-016-0558-7.pdf
- This app is tested and distributed only for the macOS system!
- Some latency is required to keep up the quality of the processed audio, so it's not recommended to use this program for watching videos.
- I haven't tested it much, but on my Mac mini it eats around 20% of CPU, which doesn't sound dramatic to me, but you are advised.

## Prerequisites

- Install BlackHole virtual device for audio loopback (separate download):
  - Download BlackHole from https://existential.audio/blackhole/
  - Open the downloaded .pkg file and follow the installation instructions.
  - Open the "Audio MIDI Setup" application on macOS.
  - Locate "BlackHole 2ch" device and set it as the default output device.
    
## Installation and Usage

- Open the image file TNMT.dmg and put the TNMT app in the macOS Applications folder.
- Run the TNMT app:
  - Select "BlackHole 2ch" as the input device and your soundcard/sound monitors as the output device
  - Click "Start Processing".
  - Enjoy your music!


## What if I don't suffer from tinnitus?

- This program seems to work well as an audio enhancer too! Try it on your favourite music and hear the difference. In this case you might want to disable the notch filtering via the related checkbox in the GUI.

## License

- This program is licensed under the GPL License. See the LICENSE file for details.
