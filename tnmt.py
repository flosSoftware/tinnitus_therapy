
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

import os
import sys
import numpy as np
import scipy
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, buttord
from scipy.signal.windows import hann
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QSlider, QLabel, QComboBox
from PyQt5.QtCore import Qt, QTimer
import threading
import pyqtgraph as pg
import logging

#from line_profiler import profile



# Determine if the script is running as a .app file or directly as a Python script
if getattr(sys, 'frozen', False):
    # Running as a .app file
    log_file_path = os.path.join(os.path.expanduser("~/Documents"), 'tnmt.log')
    logging.basicConfig(filename=log_file_path, level=logging.WARNING)
else:
    # Running directly as a Python script
    logging.basicConfig(level=logging.INFO)
    
    
    
audio_processor = None
processing_enabled = True
use_fft = False
notch_filter_enabled = True
stereo_processing_enabled = True
post_gain = 0.5
notch_filter_frequency = 13400


'''

DEPRECATED: FFT processing is not used in the current implementation    

FFT_SIZE = 2*BLOCK_SIZE  # Use a longer FFT size to reduce artifacts
LOOKAHEAD_FFT_SIZE = LOOKAHEAD_SIZE * 2  # Use a longer FFT size for the lookahead buffer

'''



    
class AudioProcessor:
    def __init__(self, post_gain, notch_filter_frequency, block_size, processing_enabled, notch_filter_enabled, stereo_processing_enabled, sampling_rate, num_bands = 12, max_freq = 18000, min_freq = 20, edge_amplification_dbs = 10):
        self.post_gain = post_gain
        
        self.processing_enabled = processing_enabled
        self.notch_filter_enabled = notch_filter_enabled
        self.stereo_processing_enabled = stereo_processing_enabled
        
        self.channels = 2 if self.stereo_processing_enabled else 1
        
        self.block_size = block_size        
        self.sampling_rate = sampling_rate
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.min_freq = min_freq

        self.hop_size = self.block_size // 2  # 50% overlap
        self.buffer_size = self.hop_size

        self.window = hann(self.block_size)[:, None] if self.stereo_processing_enabled else hann(self.block_size)  # Use a Hann window for smooth tapering

        self.nyquist = 0.5 * self.sampling_rate
        self.band_limits = np.logspace(np.log10(self.min_freq), np.log10(self.max_freq), self.num_bands + 1)
        
        half_octave_width = 1/2  # Octaves
        self.half_octave_ratio = 2 ** (half_octave_width / 2)

        # edge filter parameters
        self.edge_amplification_dbs = edge_amplification_dbs  # dB boost
        
        edge_band_octave_width = 3/8  # Octaves
        self.edge_octave_ratio = 2 ** (edge_band_octave_width/2)

        self.lookahead_blocks = 10  # Number of blocks for lookahead buffer
        self.lookahead_size = self.lookahead_blocks * self.hop_size

        self.lookahead_window = self.custom_window(self.lookahead_size)[:, None] if self.stereo_processing_enabled else self.custom_window(self.lookahead_size)

        '''
        
        DEPRECATED: FFT processing is not used in the current implementation
        
        self.lookahead_buffer = np.zeros((self.lookahead_size, self.channels))
        '''
        
        self.buffer = np.zeros((self.buffer_size, self.channels))
        self.processed_buffer = np.zeros((self.buffer_size, self.channels))
        self.lhb_filtered_bands = [np.zeros((self.lookahead_size, self.channels)) for _ in range(self.num_bands)]

        self.eq_filters = self.get_eq_filters()
        self.update_notch_edge_filters(notch_filter_frequency)

        self.spectrum_buffer = np.zeros((10, self.hop_size))
        
    


    def get_eq_filters(self):
        passband_ripple_low = 3  # dB
        stopband_attenuation_low = 30  # dB
        transition_band_pct_low = 0.3  # % transition band
        passband_ripple_middle = 1  # dB
        stopband_attenuation_middle = 40  # dB
        transition_band_pct_middle = 0.2  # % transition band
        passband_ripple_high = 3  # dB
        stopband_attenuation_high = 30  # dB
        transition_band_pct_high = 0.2  # % transition band

        filters = []
        for i in range(self.num_bands):
            filter_lowcut = self.band_limits[i] / self.nyquist
            filter_highcut = self.band_limits[i + 1] / self.nyquist
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

            ws = [filter_lowcut * (1 - transition_band_pct), filter_highcut * (1 + transition_band_pct)]  # Wider stopband for better attenuation
            if 0 < ws[0] < wp[0] < 1 and 0 < wp[1] < ws[1] < 1:
                N, Wn = buttord(wp, ws, passband_ripple, stopband_attenuation, fs=self.sampling_rate)
            else:
                raise ValueError("Invalid passband or stopband frequencies")

            sos = butter(N, Wn, btype='band', output='sos')
            filters.append(sos)

        return filters



    def update_notch_edge_filters(self, notch_filter_frequency):
        
        notch_filter_lowcut = notch_filter_frequency / self.half_octave_ratio
        notch_filter_highcut = notch_filter_frequency * self.half_octave_ratio
        
        logging.info(f"Notch filter band: {notch_filter_lowcut} Hz -> {notch_filter_highcut} Hz")
        
        low_edge_filter_lowcut = notch_filter_lowcut / self.edge_octave_ratio
        low_edge_filter_highcut = notch_filter_lowcut
        high_edge_filter_lowcut = notch_filter_highcut
        high_edge_filter_highcut = notch_filter_highcut * self.edge_octave_ratio
        
        notch_filter_lowcut = low_edge_filter_lowcut
        notch_filter_highcut = high_edge_filter_highcut
        
        try:
        
            self.notch_filter = self.get_notch_filter(notch_filter_lowcut, notch_filter_highcut)
            self.edge_filters = self.get_edge_filters(low_edge_filter_lowcut, low_edge_filter_highcut, high_edge_filter_lowcut, high_edge_filter_highcut)
        except ValueError as e:
            logging.error(f"Error updating notch filter: {e}")
            return

    def get_notch_filter(self, notch_filter_lowcut, notch_filter_highcut):
        
        notch_filter_wp = [notch_filter_lowcut / self.nyquist, notch_filter_highcut / self.nyquist]

        notch_filter_transition_band_pct = 0.2
        notch_filter_passband_ripple = 1
        notch_filter_stopband_attenuation = 50

        notch_filter_ws = [notch_filter_wp[0] * (1-notch_filter_transition_band_pct), notch_filter_wp[1] * (1+notch_filter_transition_band_pct)]  # Wider stopband for better attenuation
        # Ensure the frequencies are within the valid range
        logging.info(f"Notch filter band: {[freq * self.nyquist for freq in notch_filter_wp]} Hz -> {[freq * self.nyquist for freq in notch_filter_ws]} Hz")
        if 0 < notch_filter_ws[0] < notch_filter_wp[0] < 1 and 0 < notch_filter_wp[1] < notch_filter_ws[1] < 1:
            notch_filter_N, notch_filter_Wn = buttord(notch_filter_wp, notch_filter_ws, notch_filter_passband_ripple, notch_filter_stopband_attenuation, fs=self.sampling_rate)
        else:
            raise ValueError("Invalid passband or stopband frequencies")

        logging.info(f"Notch filter order: {notch_filter_N}, Natural frequency: {notch_filter_Wn}")

        return butter(notch_filter_N, notch_filter_Wn, btype='bandstop', output='sos')

    def get_edge_filters(self, low_edge_filter_lowcut, low_edge_filter_highcut, high_edge_filter_lowcut, high_edge_filter_highcut):
        
        edge_filter_transition_band_pct = 0.2
        edge_filter_passband_ripple = 1
        edge_filter_stopband_attenuation = 50
        
        filters = []
        
        edge_filter_wp = [low_edge_filter_lowcut / self.nyquist, low_edge_filter_highcut / self.nyquist]

        edge_filter_ws = [edge_filter_wp[0] * (1-edge_filter_transition_band_pct), edge_filter_wp[1] * (1+edge_filter_transition_band_pct)]  # Wider stopband for better attenuation
        
        # Ensure the frequencies are within the valid range
        logging.info(f"Edge filter band: {[freq * self.nyquist for freq in edge_filter_wp]} Hz -> {[freq * self.nyquist for freq in edge_filter_ws]} Hz")
        if 0 < edge_filter_ws[0] < edge_filter_wp[0] < 1 and 0 < edge_filter_wp[1] < edge_filter_ws[1] < 1:
            edge_filter_N, edge_filter_Wn = buttord(edge_filter_wp, edge_filter_ws, edge_filter_passband_ripple, edge_filter_stopband_attenuation, fs=self.sampling_rate)
        else:
            raise ValueError("Invalid passband or stopband frequencies")

        logging.info(f"Edge filter order: {edge_filter_N}, Natural frequency: {edge_filter_Wn}")

        filters.append( butter(edge_filter_N, edge_filter_Wn, btype='bandpass', output='sos') )
        
        edge_filter_wp = [high_edge_filter_lowcut / self.nyquist, high_edge_filter_highcut / self.nyquist]

        edge_filter_ws = [edge_filter_wp[0] * (1-edge_filter_transition_band_pct), edge_filter_wp[1] * (1+edge_filter_transition_band_pct)]  # Wider stopband for better attenuation
        
        # Ensure the frequencies are within the valid range
        logging.info(f"Edge filter band: {[freq * self.nyquist for freq in edge_filter_wp]} Hz -> {[freq * self.nyquist for freq in edge_filter_ws]} Hz")
        if 0 < edge_filter_ws[0] < edge_filter_wp[0] < 1 and 0 < edge_filter_wp[1] < edge_filter_ws[1] < 1:
            edge_filter_N, edge_filter_Wn = buttord(edge_filter_wp, edge_filter_ws, edge_filter_passband_ripple, edge_filter_stopband_attenuation, fs=self.sampling_rate)
        else:
            raise ValueError("Invalid passband or stopband frequencies")

        logging.info(f"Edge filter order: {edge_filter_N}, Natural frequency: {edge_filter_Wn}")

        filters.append( butter(edge_filter_N, edge_filter_Wn, btype='bandpass', output='sos') )
        
        return filters



    def custom_window(self, length, side_lobes_length=0.05):
        window = np.ones(length)
        taper_length = int(side_lobes_length * length / 2) if side_lobes_length > 0 else side_lobes_length // 2
        # Ensure side_lobe_length is not greater than half the window length
        taper_length = min(taper_length, length // 2)
        # Taper the edges using a cosine function
        taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_length)))
        window[:taper_length] = taper
        window[-taper_length:] = taper[::-1]
        
        return window
    
    
    

    #@profile
    def process_audio(self, indata, outdata, frames, time, status):
        
        if status:
            logging.info(status)
        
        # Extract the first column as a 1-dimensional array
        block = indata[:, :2] if self.stereo_processing_enabled else indata[:, 0] # block has length eq to self.hop_size!
        
        
        if self.processing_enabled:
            
            block = block - np.mean(block, axis=0) # DC offset removal

            '''
            
            DEPRECATED: FFT processing is not used in the current implementation
            
            # Update lookahead buffer
            self.lookahead_buffer = np.roll(self.lookahead_buffer, -self.hop_size, axis=0)
            self.lookahead_buffer[-self.hop_size:, :] = block
            
            '''
            
            # add non processed samples from previous block to the first part of the current block (which will be processed)
            overlapping_block = np.concatenate((self.buffer, block[:self.hop_size, :]), axis=0)
            
            # put non processed samples into the buffer
            self.buffer = block.copy()
            
            '''
            
            DEPRECATED: FFT processing is not used in the current implementation
            
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
            '''
            

            block_band_energies = np.zeros((self.num_bands, block.shape[1]))
            block_filtered_bands = np.zeros((self.num_bands, overlapping_block.shape[0], block.shape[1]))
            lhb_band_energies = np.zeros((self.num_bands, block.shape[1]))
            
            
            # Apply each bandpass filter to the new data added to the lookahead buffer
            for i, sos in enumerate(self.eq_filters):
                self.lhb_filtered_bands[i] = np.roll(self.lhb_filtered_bands[i], -self.hop_size, axis=0)
                self.lhb_filtered_bands[i][-self.hop_size:, :] = sosfiltfilt(sos, block, axis=0)
                block_filtered_band = sosfiltfilt(sos, overlapping_block, axis=0)                
                block_filtered_bands[i] = block_filtered_band
                block_band_energies[i] = np.sum(block_filtered_band**2, axis=0)
                lhb_band_energies[i] = np.sum(self.lhb_filtered_bands[i]**2, axis=0)
            

            # Compute the average energy using the lookahead buffer
            #lhb_band_energies = [np.sum(fb**2, axis=0) for fb in self.lhb_filtered_bands]
            avg_energy = np.mean(lhb_band_energies, axis=0)
            
            # Auto-equalize the spectrum
            equalized_block = np.zeros_like(overlapping_block)
            for i, filtered_band in enumerate(block_filtered_bands):
                gain = np.sqrt(avg_energy / (block_band_energies[i] + 1e-10))  # Prevent division by zero
                equalized_block += gain * filtered_band
                    
                    
            if self.notch_filter_enabled:    
                equalized_block_notched = sosfiltfilt(self.notch_filter, equalized_block, axis=0)                
                
                equalized_block_low_edge = sosfiltfilt(self.edge_filters[0], equalized_block, axis=0)
                equalized_block_high_edge = sosfiltfilt(self.edge_filters[1], equalized_block, axis=0)
                
                # Combine edge amplifications (edge_amplification_dbs dB boost)
                equalized_block = equalized_block_notched + 10**(self.edge_amplification_dbs / 20) * (equalized_block_low_edge + equalized_block_high_edge)
                    
            equalized_block *= self.post_gain
            
            # Overlap-add method (only for non-FFT processing)
            '''
            DEPRECATED: FFT processing is not used in the current implementation
            
            if not use_fft:
            '''
            equalized_block *= self.window
                        
            # Overlap-add method
            equalized_block[:self.hop_size, :] += self.processed_buffer
            self.processed_buffer = equalized_block[self.hop_size:, :].copy() 
            
            out_block = equalized_block[:self.hop_size, :]       
            
            
            # Copy the processed audio to the output
            outdata[:] = out_block.reshape(-1,  self.channels)
            
            # Update the spectrum buffer
            self.spectrum_buffer = np.roll(self.spectrum_buffer, -1, axis=0)
            self.spectrum_buffer[-1, :] = out_block[:, 0]  # Use only the left channel for spectrum analysis
        
        else:
            outdata[:] = block.reshape(-1, self.channels)
            # Update the spectrum buffer
            self.spectrum_buffer = np.roll(self.spectrum_buffer, -1, axis=0)
            self.spectrum_buffer[-1, :] = block[:, 0]  # Use only the left channel for spectrum analysis
    
    

    def compute_average_spectrum(self):
        if np.all(self.spectrum_buffer == 0):
            return np.zeros(self.hop_size // 2 + 1)
        avg_spectrum = np.mean(np.abs(scipy.fft.rfft(self.spectrum_buffer, axis=1)), axis=0)
        return avg_spectrum


    
    '''
    DEPRECATED: FFT processing is not used in the current implementation


    def toggle_use_fft(state):
        global use_fft
        use_fft = state == 2
        logging.info(f"FFT equalizer enabled: {use_fft}") 
    '''
        
    
class AudioThread(threading.Thread):
    def __init__(self, input_device, output_device):
        super().__init__()
        self.stream = None
        self.running = False
        self.input_device = input_device
        self.output_device = output_device

    def run(self):
        global audio_processor
        try:
            self.running = True
            self.stream = sd.Stream(callback=self.audio_callback, channels=audio_processor.channels, samplerate=audio_processor.sampling_rate, blocksize=audio_processor.hop_size, device=(self.input_device, self.output_device))
            self.stream.start()
            while self.running:
                sd.sleep(100)
            self.stream.stop()
            self.stream.close()
        except Exception as e:
            logging.error(f"Error processing audio stream: {e}")

    def stop(self):
        self.running = False

    def audio_callback(self, indata, outdata, frames, time, status):
        try:
            if status:
                print(status)
            global audio_processor
            audio_processor.process_audio(indata, outdata, frames, time, status)
        except Exception as e:
            logging.error(f"Error processing audio stream: {e}")
    
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
        
        self.block_size_label = QLabel('Select Block Size:')
        layout.addWidget(self.block_size_label)
        
        self.block_size_combo = QComboBox()
        self.block_size_combo.addItems(['16384', '32768', '65536'])
        self.block_size_combo.setCurrentText('32768')
        layout.addWidget(self.block_size_combo)
        
        self.start_button = QPushButton('Start Processing')
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)
        
        self.setLayout(layout)
        
    def start_processing(self):

        input_device = self.input_device_combo.currentIndex()
        output_device = self.output_device_combo.currentIndex()
        block_size = int(self.block_size_combo.currentText())        
        
        global audio_processor, post_gain, notch_filter_frequency, processing_enabled, notch_filter_enabled, stereo_processing_enabled
        devices = sd.query_devices()
        sampling_rate = devices[input_device]['default_samplerate']
        audio_processor = AudioProcessor(
            post_gain=post_gain,
            notch_filter_frequency=notch_filter_frequency,
            block_size=block_size,
            processing_enabled=processing_enabled,
            notch_filter_enabled=notch_filter_enabled,
            stereo_processing_enabled=True,
            sampling_rate=sampling_rate,
            num_bands=12,
            max_freq=18000,
            min_freq=20,
            edge_amplification_dbs=10
        )
        
        
        self.hide()
        self.main_window = AudioProcessingApp(input_device, output_device)
        self.main_window.show()
     
        

class AudioProcessingApp(QWidget):
    def __init__(self, input_device, output_device):

        super().__init__()
        self.spectrogram_enabled = True
        self.input_device = input_device
        self.output_device = output_device
        
        self.audio_thread = AudioThread(self.input_device, self.output_device)
        self.initUI()
        self.start_audio()

    def initUI(self):
        layout = QVBoxLayout()
        
        self.processing_checkbox = QCheckBox("Enable processing")   
        self.processing_checkbox.setChecked(processing_enabled)
        self.processing_checkbox.stateChanged.connect(self.toggle_processing)
        layout.addWidget(self.processing_checkbox)
        
        self.spectrogram_checkbox = QCheckBox("Enable spectrogram")
        self.spectrogram_checkbox.setChecked(True)
        self.spectrogram_checkbox.stateChanged.connect(self.toggle_spectrogram)
        layout.addWidget(self.spectrogram_checkbox)
        
        
        
        '''
        
        DEPRECATED: FFT processing is not used in the current implementation

        self.use_fft_checkbox = QCheckBox("Enable FFT equalizer")
        self.use_fft_checkbox.setChecked(use_fft)
        self.use_fft_checkbox.stateChanged.connect(toggle_use_fft)
        layout.addWidget(self.use_fft_checkbox)
        
        '''
        self.notch_filter_checkbox = QCheckBox("Enable notch filter")
        self.notch_filter_checkbox.setChecked(notch_filter_enabled)
        self.notch_filter_checkbox.stateChanged.connect(self.toggle_notch_filter)
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
        self.timer.start(500)
        
        self.setWindowTitle("TNMT Audio Processor")
        self.show()
        
    def start_audio(self):
        self.audio_thread.start()
        
    def toggle_notch_filter(self, state):
        global notch_filter_enabled, audio_processor
        notch_filter_enabled = state == 2
        logging.info(f"Notch filter enabled: {notch_filter_enabled}")
        audio_processor.notch_filter_enabled = notch_filter_enabled
        
    def toggle_processing(self, state):
        global processing_enabled, audio_processor
        processing_enabled = state == 2
        logging.info(f"Processing enabled: {processing_enabled}")
        audio_processor.processing_enabled = processing_enabled
        
    def update_notch_filter_frequency_label(self, value):
        global notch_filter_frequency, audio_processor
        notch_filter_frequency = value
        logging.info(f"Frequency set to: {value} Hz")
        self.notch_filter_frequency_label.setText(f"{value} Hz")
        audio_processor.update_notch_edge_filters(value)
        
    def update_post_gain_label(self, value):
        global post_gain, audio_processor
        post_gain = value / 10.0  # Convert slider value to gain
        self.post_gain_label.setText(f"{post_gain}")
        logging.info(f"Post gain set to: {post_gain}")
        audio_processor.post_gain = post_gain       
        
    def toggle_spectrogram(self, state):
        self.spectrogram_enabled = state == 2
        if self.spectrogram_enabled:
            self.timer.start(500)
        else:
            self.timer.stop()
            self.plot_widget.clear()

    def update_plot(self):
        global audio_processor

        avg_spectrum = audio_processor.compute_average_spectrum()
        freqs = np.fft.rfftfreq(audio_processor.hop_size, 1 / audio_processor.sampling_rate)      
        
        self.plot_widget.clear()
        self.plot_widget.plot(freqs, 20 * np.log10(avg_spectrum + 1e-10))  # Plot in dB
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLabel('left', 'Magnitude', units='dB')
        self.plot_widget.setTitle('Average Spectrum')
        self.plot_widget.setXRange(0, 20000)
        self.plot_widget.setYRange(-50, 50)

            
            

    def closeEvent(self, event):
        self.audio_thread.stop()
        self.audio_thread.join()  # Wait for the audio thread to finish
        event.accept()


# Start the GUI application in the main thread
app = QApplication(sys.argv)
device_selection_window = DeviceSelectionWindow()
device_selection_window.show()
sys.exit(app.exec_())