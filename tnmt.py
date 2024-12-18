import sys
import numpy as np
import scipy
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, buttord
from scipy.signal.windows import hann
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QSlider, QLabel, QComboBox
from PyQt5.QtCore import Qt
import threading
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

'''
TNMT Audio Processor is an implementation of the audio processing pipeline described in the article "Clinical trial on tonal tinnitus with tailor-made notched music training" by Pantev et al. (2016). The pipeline includes a multi-band auto equalizer, a notch filter around tinnitus frequency, and edge amplification at the notch filter boundaries. 

Copyright (C) 2024  Alberto Fiore

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

For any questions or suggestions, please contact the author at flos.software@gmail.com.
''' 


def custom_window(length, side_lobes_length=0.05):
    window = np.ones(length)
    taper_length = int(side_lobes_length * length / 2) if side_lobes_length > 0 else side_lobes_length // 2
    # Ensure side_lobe_length is not greater than half the window length
    taper_length = min(taper_length, length // 2)
    # Taper the edges using a cosine function
    taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_length)))
    window[:taper_length] = taper
    window[-taper_length:] = taper[::-1]
    
    return window



def get_notch_filter():
    global notch_filter_lowcut, notch_filter_highcut, SAMPLING_RATE, nyquist
    
    notch_filter_wp = [notch_filter_lowcut / nyquist, notch_filter_highcut / nyquist]

    notch_filter_transition_band_pct = 0.2
    notch_filter_passband_ripple = 1
    notch_filter_stopband_attenuation = 50

    notch_filter_ws = [notch_filter_wp[0] * (1-notch_filter_transition_band_pct), notch_filter_wp[1] * (1+notch_filter_transition_band_pct)]  # Wider stopband for better attenuation
    # Ensure the frequencies are within the valid range
    print(f"Notch filter band: {[freq * nyquist for freq in notch_filter_wp]} Hz -> {[freq * nyquist for freq in notch_filter_ws]} Hz")
    if 0 < notch_filter_ws[0] < notch_filter_wp[0] < 1 and 0 < notch_filter_wp[1] < notch_filter_ws[1] < 1:
        notch_filter_N, notch_filter_Wn = buttord(notch_filter_wp, notch_filter_ws, notch_filter_passband_ripple, notch_filter_stopband_attenuation, fs=SAMPLING_RATE)
    else:
        raise ValueError("Invalid passband or stopband frequencies")

    print(f"Notch filter order: {notch_filter_N}, Natural frequency: {notch_filter_Wn}")

    return butter(notch_filter_N, notch_filter_Wn, btype='bandstop', output='sos')

def get_edge_filters():
    global low_edge_filter_lowcut, low_edge_filter_highcut, high_edge_filter_lowcut, high_edge_filter_highcut, SAMPLING_RATE, nyquist
    
    edge_filter_transition_band_pct = 0.2
    edge_filter_passband_ripple = 1
    edge_filter_stopband_attenuation = 50
    
    filters = []
    
    edge_filter_wp = [low_edge_filter_lowcut / nyquist, low_edge_filter_highcut / nyquist]

    edge_filter_ws = [edge_filter_wp[0] * (1-edge_filter_transition_band_pct), edge_filter_wp[1] * (1+edge_filter_transition_band_pct)]  # Wider stopband for better attenuation
    
    # Ensure the frequencies are within the valid range
    print(f"Edge filter band: {[freq * nyquist for freq in edge_filter_wp]} Hz -> {[freq * nyquist for freq in edge_filter_ws]} Hz")
    if 0 < edge_filter_ws[0] < edge_filter_wp[0] < 1 and 0 < edge_filter_wp[1] < edge_filter_ws[1] < 1:
        edge_filter_N, edge_filter_Wn = buttord(edge_filter_wp, edge_filter_ws, edge_filter_passband_ripple, edge_filter_stopband_attenuation, fs=SAMPLING_RATE)
    else:
        raise ValueError("Invalid passband or stopband frequencies")

    print(f"Edge filter order: {edge_filter_N}, Natural frequency: {edge_filter_Wn}")

    filters.append( butter(edge_filter_N, edge_filter_Wn, btype='bandpass', output='sos') )
    
    edge_filter_wp = [high_edge_filter_lowcut / nyquist, high_edge_filter_highcut / nyquist]

    edge_filter_ws = [edge_filter_wp[0] * (1-edge_filter_transition_band_pct), edge_filter_wp[1] * (1+edge_filter_transition_band_pct)]  # Wider stopband for better attenuation
    
    # Ensure the frequencies are within the valid range
    print(f"Edge filter band: {[freq * nyquist for freq in edge_filter_wp]} Hz -> {[freq * nyquist for freq in edge_filter_ws]} Hz")
    if 0 < edge_filter_ws[0] < edge_filter_wp[0] < 1 and 0 < edge_filter_wp[1] < edge_filter_ws[1] < 1:
        edge_filter_N, edge_filter_Wn = buttord(edge_filter_wp, edge_filter_ws, edge_filter_passband_ripple, edge_filter_stopband_attenuation, fs=SAMPLING_RATE)
    else:
        raise ValueError("Invalid passband or stopband frequencies")

    print(f"Edge filter order: {edge_filter_N}, Natural frequency: {edge_filter_Wn}")

    filters.append( butter(edge_filter_N, edge_filter_Wn, btype='bandpass', output='sos') )
    
    return filters


stop_flag = threading.Event()  # Flag to signal threads to stop


processing_enabled = True
use_fft = False
notch_filter_enabled = True

stereo_processing_enabled = True

# Define block size and overlap
BLOCK_SIZE = 4096 * 8  # SIZE OF A NEW BLOCK OF SAMPLES..
HOP_SIZE = BLOCK_SIZE // 2  # 50% overlap
BUFFER_SIZE = HOP_SIZE  # SIZE OF THE BUFFER
FFT_SIZE = 2*BLOCK_SIZE  # Use a longer FFT size to reduce artifacts

window = hann(BLOCK_SIZE)[:, None] if stereo_processing_enabled else hann(BLOCK_SIZE)  # Use a Hann window for smooth tapering


# Initialize global variables
SAMPLING_RATE = 48000  # Sampling rate
CHANNELS = 2 if stereo_processing_enabled else 1  

# Define the number of bands and the frequency range
NUM_BANDS = 12
MAX_FREQ = 18000  # Maximum frequency to consider for equalization
MIN_FREQ = 20  # Minimum frequency to consider for equalization
nyquist = 0.5 * SAMPLING_RATE
#band_limits = np.linspace(MIN_FREQ, MAX_FREQ, NUM_BANDS + 1)
band_limits = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), NUM_BANDS + 1)

# Initialize global variables for lookahead buffer
LOOKAHEAD_BLOCKS = 10  # Number of blocks for lookahead buffer
LOOKAHEAD_SIZE = LOOKAHEAD_BLOCKS * HOP_SIZE
LOOKAHEAD_FFT_SIZE = LOOKAHEAD_SIZE * 2  # Use a longer FFT size for the lookahead buffer
lookahead_window = custom_window(LOOKAHEAD_SIZE)[:, None] if stereo_processing_enabled else custom_window(LOOKAHEAD_SIZE) #hann(LOOKAHEAD_SIZE)

lookahead_buffer = np.zeros((LOOKAHEAD_SIZE,CHANNELS))
buffer = np.zeros((BUFFER_SIZE,CHANNELS))
processed_buffer = np.zeros((BUFFER_SIZE,CHANNELS))
lhb_filtered_bands = [np.zeros((LOOKAHEAD_SIZE, CHANNELS)) for _ in range(NUM_BANDS)]  # Initialize filtered bands for lookahead buffer


post_gain = 0.5


# Define passband and stopband requirements
passband_ripple_low = 3  # dB
stopband_attenuation_low = 30  # dB
transition_band_pct_low = 0.3  # % transition band
passband_ripple_middle = 1  # dB
stopband_attenuation_middle = 40  # dB
transition_band_pct_middle = 0.2  # % transition band
passband_ripple_high = 3  # dB
stopband_attenuation_high = 30  # dB
transition_band_pct_high = 0.2  # % transition band

# Design an array of Butterworth bandpass filters using buttord
sos_filters = []
for i in range(NUM_BANDS):
    filter_lowcut = band_limits[i] / nyquist
    filter_highcut = band_limits[i + 1] / nyquist
    wp = [filter_lowcut, filter_highcut]
    if filter_lowcut < 0.025:
        transition_band_pct = transition_band_pct_low
        passband_ripple = passband_ripple_low
        stopband_attenuation = stopband_attenuation_low
    elif filter_lowcut > 0.1:
        transition_band_pct = transition_band_pct_low
        passband_ripple = passband_ripple_low
        stopband_attenuation = stopband_attenuation_low
    else:
        transition_band_pct = transition_band_pct_middle
        passband_ripple = passband_ripple_middle
        stopband_attenuation = stopband_attenuation_middle
        
    ws = [filter_lowcut * (1-transition_band_pct), filter_highcut * (1+transition_band_pct)]  # Wider stopband for better attenuation
    # Ensure the frequencies are within the valid range
    print(f"Band {i}: {[freq * nyquist for freq in wp]} Hz -> {[freq * nyquist for freq in ws]} Hz")
    if 0 < ws[0] < wp[0] < 1 and 0 < wp[1] < ws[1] < 1:
        N, Wn = buttord(wp, ws, passband_ripple, stopband_attenuation, fs=SAMPLING_RATE)
    else:
        raise ValueError("Invalid passband or stopband frequencies")

    print(f"Filter order: {N}, Natural frequency: {Wn}")

    sos = butter(N, Wn, btype='band', output='sos')
    sos_filters.append(sos)

# notch filter parameters
notch_filter_frequency = 13400
half_octave_width = 1/2  # Octaves
half_octave_ratio = 2 ** (half_octave_width / 2)
notch_filter_lowcut = notch_filter_frequency / half_octave_ratio
notch_filter_highcut = notch_filter_frequency * half_octave_ratio

print(f"Notch filter band: {notch_filter_lowcut} Hz -> {notch_filter_highcut} Hz")

# edge filter parameters
edge_amplification_dbs = 10  # dB boost
edge_band_octave_width = 3/8  # Octaves
edge_octave_ratio = 2 ** (edge_band_octave_width/2)
low_edge_filter_lowcut = notch_filter_lowcut / edge_octave_ratio
low_edge_filter_highcut = notch_filter_lowcut
high_edge_filter_lowcut = notch_filter_highcut
high_edge_filter_highcut = notch_filter_highcut * edge_octave_ratio


notch_filter_lowcut = low_edge_filter_lowcut
notch_filter_highcut = high_edge_filter_highcut


notch_filter = get_notch_filter()
edge_filters = get_edge_filters()

spectrum_buffer = np.zeros((10, HOP_SIZE))


def audio_callback(indata, outdata, frames, time, status):
    
    global lookahead_buffer, HOP_SIZE,  buffer, processed_buffer, window, lookahead_window
    global FFT_SIZE, NUM_BANDS, band_limits, lhb_filtered_bands, sos_filters, SAMPLING_RATE, CHANNELS
    global use_fft, processing_enabled, notch_filter_enabled, spectrum_buffer
    global notch_filter_lowcut, notch_filter_highcut, notch_filter, low_edge_filter_lowcut, low_edge_filter_highcut, high_edge_filter_lowcut, high_edge_filter_highcut, edge_amplification_dbs, edge_filters 
    global post_gain, stereo_processing_enabled
    
    
    
    if status:
        print(status)
    
    # Extract the first column as a 1-dimensional array
    block = indata[:, :2] if stereo_processing_enabled else indata[:, 0]
    
    
    if processing_enabled:
        
        block = block - np.mean(block, axis=0)

        # Update lookahead buffer
        lookahead_buffer = np.roll(lookahead_buffer, -HOP_SIZE, axis=0)
        lookahead_buffer[-HOP_SIZE:, :] = block
        
        # add non processed samples from previous block to the first part of the current block (which will be processed)
        overlapping_block = np.concatenate((buffer,block[:HOP_SIZE, :]), axis=0)
        
        # put non processed samples into the buffer
        buffer = block.copy()
        
        
        if use_fft:
            # Apply window to the overlapping block
            windowed_block = overlapping_block * window
            
            # Apply window to the lookahead buffer
            windowed_lookahead_buffer = lookahead_buffer * lookahead_window
            
            # FFT processing to calculate average energy in each band using the lookahead buffer
            lookahead_fft = scipy.fft.rfft(windowed_lookahead_buffer, n=LOOKAHEAD_FFT_SIZE, axis=0)
            freqs = scipy.fft.rfftfreq(LOOKAHEAD_FFT_SIZE, 1 / SAMPLING_RATE)
            band_energies = np.zeros((NUM_BANDS, windowed_block.shape[1]))
            
            for i in range(NUM_BANDS):
                band_mask = (freqs >= band_limits[i]) & (freqs < band_limits[i + 1])
                band_energy = np.sum(np.abs(lookahead_fft[band_mask])**2, axis=0)
                band_energies[i] = band_energy
                
            avg_energy = np.mean(band_energies, axis=0)
            
            # FFT of the windowed block with zero-padding
            block_fft = scipy.fft.rfft(windowed_block, n=FFT_SIZE, axis=0)
            freqs = scipy.fft.rfftfreq(FFT_SIZE, 1 / SAMPLING_RATE)
            
            # Auto-equalize the spectrum of the block using the average band energy
            for i in range(NUM_BANDS):
                band_mask = (freqs >= band_limits[i]) & (freqs < band_limits[i + 1])
                block_band_energy = np.sum(np.abs(block_fft[band_mask])**2, axis=0)
                gain = np.sqrt(avg_energy / (block_band_energy + 1e-10))
                block_fft[band_mask] *= gain
                    

            # Inverse FFT to get the filtered block
            equalized_block = scipy.fft.irfft(block_fft, n=FFT_SIZE, axis=0)
            
            # Extract the portion corresponding to the original block
            equalized_block = equalized_block[:2*HOP_SIZE, :]
            
    
            
        else:
        
            # Apply each bandpass filter to the new data added to the lookahead buffer
            for i, sos in enumerate(sos_filters):
                lhb_filtered_bands[i] = np.roll(lhb_filtered_bands[i], -HOP_SIZE, axis=0)
                lhb_filtered_bands[i][-HOP_SIZE:, :] = sosfiltfilt(sos, block, axis=0)
            
            # Apply each bandpass filter to the block
            block_band_energies = np.zeros((NUM_BANDS, block.shape[1]))
            block_filtered_bands = np.zeros((NUM_BANDS, overlapping_block.shape[0], block.shape[1]))

            for i, sos in enumerate(sos_filters):
                block_filtered_band = sosfiltfilt(sos, overlapping_block, axis=0)                
                block_filtered_bands[i] = block_filtered_band
                block_band_energies[i] = np.sum(block_filtered_band**2, axis=0)
                
            
            # Compute the average energy using the lookahead buffer
            lhb_band_energies = [np.sum(fb**2, axis=0) for fb in lhb_filtered_bands]
            avg_energy = np.mean(lhb_band_energies, axis=0)
            
            # Auto-equalize the spectrum
            equalized_block = np.zeros_like(overlapping_block)
            for i, filtered_band in enumerate(block_filtered_bands):
                gain = np.sqrt(avg_energy / (block_band_energies[i] + 1e-10))  # Prevent division by zero
                equalized_block += gain * filtered_band
                
                
        if notch_filter_enabled:    
            equalized_block_notched = sosfiltfilt(notch_filter, equalized_block, axis=0)                
            
            equalized_block_low_edge = sosfiltfilt(edge_filters[0], equalized_block, axis=0)
            equalized_block_high_edge = sosfiltfilt(edge_filters[1], equalized_block, axis=0)
            
            # Combine edge amplifications (edge_amplification_dbs dB boost)
            equalized_block = equalized_block_notched + 10**(edge_amplification_dbs / 20) * (equalized_block_low_edge + equalized_block_high_edge)
                
        equalized_block *= post_gain
        
        # Overlap-add method (only for non-FFT processing)
        if not use_fft:
            equalized_block *= window
                    
        # Overlap-add method
        equalized_block[:HOP_SIZE, :] += processed_buffer
        processed_buffer = equalized_block[HOP_SIZE:, :].copy() 
        
        out_block = equalized_block[:HOP_SIZE, :]       
        
        
        # Copy the processed audio to the output
        outdata[:] = out_block.reshape(-1,  2 if stereo_processing_enabled else 1)
        
        # Update the spectrum buffer
        spectrum_buffer = np.roll(spectrum_buffer, -1, axis=0)
        spectrum_buffer[-1, :] = out_block[:, 0]  # Use only the left channel for spectrum analysis
    
    else:
        outdata[:] = block.reshape(-1, 2 if stereo_processing_enabled else 1)
        # Update the spectrum buffer
        spectrum_buffer = np.roll(spectrum_buffer, -1, axis=0)
        spectrum_buffer[-1, :] = block[:, 0]  # Use only the left channel for spectrum analysis
    
    

def compute_average_spectrum():
    if np.all(spectrum_buffer == 0):
        return np.zeros(HOP_SIZE // 2 + 1)
    avg_spectrum = np.mean(np.abs(scipy.fft.rfft(spectrum_buffer, axis=1)), axis=0)
    return avg_spectrum

# Start audio stream in a separate thread
def start_audio():
    global input_device, output_device
    
    
    stream = sd.Stream(channels=CHANNELS, samplerate=SAMPLING_RATE, blocksize=HOP_SIZE,  # Increased blocksize
                       dtype='float32', callback=audio_callback,
                       device=(input_device, output_device))
    with stream:
        print("Processing audio... Adjust sliders to change parameters.")
        while not stop_flag.is_set():
            sd.sleep(100)  # Sleep for a short period to check the stop flag

# Functions to update processing states
def toggle_notch_filter(state):
    global notch_filter_enabled
    notch_filter_enabled = state == 2
    print(f"Notch filter enabled: {notch_filter_enabled}")
    
def toggle_processing(state):
    global processing_enabled
    processing_enabled = state == 2
    print(f"Processing enabled: {processing_enabled}")

def toggle_use_fft(state):
    global use_fft
    use_fft = state == 2
    print(f"FFT equalizer enabled: {use_fft}") 
    
    
def update_notch_filter_params():
    global notch_filter_frequency, notch_filter_lowcut, notch_filter_highcut, low_edge_filter_lowcut, low_edge_filter_highcut, high_edge_filter_lowcut, high_edge_filter_highcut, edge_octave_ratio, half_octave_ratio
    global notch_filter, edge_filters
    
    notch_filter_lowcut = notch_filter_frequency / half_octave_ratio
    notch_filter_highcut = notch_filter_frequency * half_octave_ratio
    
    print(f"Notch filter band: {notch_filter_lowcut} Hz -> {notch_filter_highcut} Hz")
    
    low_edge_filter_lowcut = notch_filter_lowcut / edge_octave_ratio
    low_edge_filter_highcut = notch_filter_lowcut
    high_edge_filter_lowcut = notch_filter_highcut
    high_edge_filter_highcut = notch_filter_highcut * edge_octave_ratio
    
    notch_filter_lowcut = low_edge_filter_lowcut
    notch_filter_highcut = high_edge_filter_highcut
    
    notch_filter = get_notch_filter()
    edge_filters = get_edge_filters()

    
    
class DeviceSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Select Input and Output Devices')
        
        layout = QVBoxLayout()
        
        self.input_device_label = QLabel('Select Input Device:')
        layout.addWidget(self.input_device_label)
        
        self.input_device_combo = QComboBox()
        self.input_device_combo.addItems([device['name'] for device in sd.query_devices()])
        layout.addWidget(self.input_device_combo)
        
        self.output_device_label = QLabel('Select Output Device:')
        layout.addWidget(self.output_device_label)
        
        self.output_device_combo = QComboBox()
        self.output_device_combo.addItems([device['name'] for device in sd.query_devices()])
        layout.addWidget(self.output_device_combo)
        
        self.start_button = QPushButton('Start Processing')
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)
        
        self.setLayout(layout)
        
    def start_processing(self):
        global input_device, output_device
        input_device = self.input_device_combo.currentIndex()
        output_device = self.output_device_combo.currentIndex()
        
        self.hide()
        self.main_window = AudioProcessingApp()
        self.main_window.show()
        
        # Start audio processing in a separate thread
        global audio_thread
        audio_thread = threading.Thread(target=start_audio)
        audio_thread.start()

class AudioProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        self.processing_checkbox = QCheckBox("Enable processing")   
        self.processing_checkbox.setChecked(processing_enabled)
        self.processing_checkbox.stateChanged.connect(toggle_processing)
        layout.addWidget(self.processing_checkbox)
        
        self.use_fft_checkbox = QCheckBox("Enable FFT equalizer")
        self.use_fft_checkbox.setChecked(use_fft)
        self.use_fft_checkbox.stateChanged.connect(toggle_use_fft)
        layout.addWidget(self.use_fft_checkbox)
        
        self.notch_filter_checkbox = QCheckBox("Enable notch filter")
        self.notch_filter_checkbox.setChecked(notch_filter_enabled)
        self.notch_filter_checkbox.stateChanged.connect(toggle_notch_filter)
        layout.addWidget(self.notch_filter_checkbox)
        
        
        notch_layout = QHBoxLayout()
        self.notch_filter_frequency_slider = QSlider(Qt.Horizontal)
        self.notch_filter_frequency_slider.setRange(1000, 14000)
        self.notch_filter_frequency_slider.setValue(notch_filter_frequency)
        self.notch_filter_frequency_slider.setTickPosition(QSlider.TicksBelow)
        self.notch_filter_frequency_slider.setTickInterval(1000)
        self.notch_filter_frequency_slider.valueChanged.connect(self.update_notch_filter_frequency_label)
        notch_label = QLabel("Filter Frequency")
        notch_label.setFixedWidth(100)  # Set a fixed width for alignment
        notch_layout.addWidget(notch_label)
        notch_layout.addWidget(self.notch_filter_frequency_slider)
        self.notch_filter_frequency_label = QLabel(f"{notch_filter_frequency} Hz")
        notch_layout.addWidget(self.notch_filter_frequency_label)
        layout.addLayout(notch_layout)
        
        post_gain_layout = QHBoxLayout()
        self.post_gain_slider = QSlider(Qt.Horizontal)
        self.post_gain_slider.setRange(1, 20)
        self.post_gain_slider.setValue(int(post_gain * 10))
        self.post_gain_slider.setTickPosition(QSlider.TicksBelow)
        self.post_gain_slider.setTickInterval(1)
        self.post_gain_slider.valueChanged.connect(self.update_post_gain_label)
        post_gain_label = QLabel("Post Gain")
        post_gain_label.setFixedWidth(100)  # Set a fixed width for alignment
        post_gain_layout.addWidget(post_gain_label)
        post_gain_layout.addWidget(self.post_gain_slider)
        self.post_gain_label = QLabel(f"{post_gain}")
        post_gain_layout.addWidget(self.post_gain_label)
        layout.addLayout(post_gain_layout)
        
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(500)  # Update every 100 ms
        
        self.setWindowTitle("TNMT Audio Processor")
        self.show()
        
    def update_notch_filter_frequency_label(self, value):
        global notch_filter_frequency
        notch_filter_frequency = value
        print(f"Frequency set to: {value} Hz")
        self.notch_filter_frequency_label.setText(f"{value} Hz")
        update_notch_filter_params()
        
    def update_post_gain_label(self, value):
        global post_gain
        post_gain = value / 10.0  # Convert slider value to gain
        self.post_gain_label.setText(f"{post_gain}")
        print(f"Post gain set to: {post_gain}")       
        
    def update_plot(self):
        global HOP_SIZE, SAMPLING_RATE, MAX_FREQ
        avg_spectrum = compute_average_spectrum()
        freqs = np.fft.rfftfreq(HOP_SIZE, 1 / SAMPLING_RATE)      
        
        self.plot_widget.clear()
        self.plot_widget.plot(freqs, 20 * np.log10(avg_spectrum + 1e-10))  # Plot in dB
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLabel('left', 'Magnitude', units='dB')
        self.plot_widget.setTitle('Average Spectrum')
        self.plot_widget.setXRange(0, 20000)
        self.plot_widget.setYRange(-50, 50)

    def closeEvent(self, event):
        stop_flag.set()  # Signal the threads to stop
        audio_thread.join()  # Wait for the audio thread to finish
        event.accept()


# Start the GUI application in the main thread
app = QApplication(sys.argv)
device_selection_window = DeviceSelectionWindow()
device_selection_window.show()
sys.exit(app.exec_())