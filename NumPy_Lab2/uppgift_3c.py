import numpy as np
import matplotlib.pyplot as plt
import scipy

data_audio_file = scipy.io.wavfile.read('Cdur.wav')
F = scipy.fftpack.fft(data_audio_file[1])

# Get the number of samples in the audio file
num_samples = len(F)

# Calculate the frequency range corresponding to the FFT
sampling_rate = data_audio_file[0]
freq_range = np.fft.fftfreq(num_samples, d=1/sampling_rate)

plt.plot(freq_range, np.abs(F), 'r', label='Fourier transform')

# Set the x-axis limits from 0 to 20000 Hz
plt.xlim(0, 2000)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of Audio Signal')
plt.show()


def frequency_detection(audio_file, freq_range):
    F = scipy.fftpack.fft(audio_file[1])
    num_samples = len(F)
    sampling_rate = audio_file[0]
    freq_bins = np.fft.fftfreq(num_samples, d=1/sampling_rate)
    mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])
    F_filtered = F[mask]
    freq_bins_filtered = freq_bins[mask]
    max_freq = freq_bins_filtered[np.argmax(np.abs(F_filtered))]
    return max_freq


def overtones(data_audio_file):
    e_overtones = []
    freq_range = (320, 340)
    max_freq = frequency_detection(data_audio_file, freq_range)
    if max_freq:
        for i in range(1, 10):
            overtone_freq = max_freq * i
            print(f"Overtone {i}: {overtone_freq}")
            e_overtones.append(overtone_freq)
    else:
        print("Couldn't find overtone")
    return e_overtones


d_sharp_overtones = overtones(data_audio_file)


def major_to_minor(overtones_list):
    new_tones_list = []
    for tone in overtones_list:
        new_tone = tone * 0.944
        print(f"Major tone: {tone}, Minor tone: {new_tone}")
        new_tones_list.append(new_tone)
    return new_tones_list


def shift_freq(data_audio_file, freq_range, target_note):
    # Find the frequency of the note to be shifted
    base_freq = frequency_detection(data_audio_file, freq_range)

    # Calculate the ratio for frequency shifting
    freq_ratio = target_note / base_freq

    # Perform frequency shifting
    audio_data = data_audio_file[1]
    num_samples = len(audio_data)
    F = scipy.fftpack.fft(audio_data)
    freq_bins = np.fft.fftfreq(num_samples, d=1 / data_audio_file[0])
    F_shifted = F * freq_ratio
    audio_data_shifted = np.real(scipy.fftpack.ifft(F_shifted))

    return data_audio_file[0], audio_data_shifted


def save_wav(audio_file, filename, dtype=np.int16):
    audio_data = audio_file[1].astype(dtype)
    audio_data_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    scipy.io.wavfile.write(filename, audio_file[0], audio_data_normalized)


# Example usage:
# Load your audio file
sampling_rate, audio_data = scipy.io.wavfile.read('Cdur.wav')
audio_file = (sampling_rate, audio_data)
freq_range_E = (320, 340)

# Define the target frequency for D flat
target_note_D_flat = 288.665

# Shift frequencies corresponding to E to match D flat
audio_file_shifted = shift_freq(audio_file, freq_range_E, d_sharp_overtones[0])

# Save the modified audio to a new file
save_wav(audio_file_shifted, 'Cmin.wav')

