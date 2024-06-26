import numpy as np
import matplotlib.pyplot as plt
import scipy

# Uppgift 3 a

audio_file = scipy.io.wavfile.read('Piano_1_C.wav')
F = scipy.fftpack.fft(audio_file[1])

# Get the number of samples in the audio file
num_samples = len(F)

# Calculate the frequency range corresponding to the FFT
sampling_rate = audio_file[0]
freq_range = np.fft.fftfreq(num_samples, d=1/sampling_rate)

plt.plot(freq_range, np.abs(F), 'r', label='Fourier transform')

# Set the x-axis limits from 0 to 20000 Hz
plt.xlim(0, 2000)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of Audio Signal')
plt.show()

def frequency_detection(audio_file):
    F = scipy.fftpack.fft(audio_file[1])
    num_samples = len(F)
    sampling_rate = audio_file[0]
    freq_range = np.fft.fftfreq(num_samples, d=1/sampling_rate)
    max_freq = freq_range[np.argmax(np.abs(F))]
    return max_freq

# Uppgift 3 b

"""def pitch_detection(audio_file):
    F = scipy.fftpack.fft(audio_file[1])
    num_samples = len(F)
    sampling_rate = audio_file[0]
    freq_range = np.fft.fftfreq(num_samples, d=1/sampling_rate)
    max_freq = freq_range[np.argmax(np.abs(F))]
    if max_freq == ufloat(261.6256, 5):
        return max_freq, "C"
    elif max_freq == ufloat(293.6648, 5):
        return max_freq, "D"
    elif max_freq == ufloat(329.6276, 5):
        return max_freq, "E"
    elif max_freq == ufloat(349.2282, 5):
        return max_freq, "F"
    elif max_freq == ufloat(391.9954, 5):
        return max_freq, "G"
    elif max_freq == ufloat(440.0000, 5):
        return max_freq, "A"
    else:
        return max_freq, "Couldn't find note"""


def get_pitch(audio_file):
    F = scipy.fftpack.fft(audio_file[1])
    num_samples = len(F)
    sampling_rate = audio_file[0]
    freq_range = np.fft.fftfreq(num_samples, d=1 / sampling_rate)
    max_freq = freq_range[np.argmax(np.abs(F))]

    while max_freq > 500:
        max_freq = max_freq / 2

    if 261.6256 - 5 <= max_freq <= 261.6256 + 5:
        return "Frekvens:", max_freq, "C"
    elif 277.1826 - 5 <= max_freq <= 277.1826 + 5:
        return "Frekvens:", max_freq, "C#"
    elif 293.6648 - 5 <= max_freq <= 293.6648 + 5:
        return "Frekvens:", max_freq, "D"
    elif 311.1270 - 5 <= max_freq <= 311.1270 + 5:
        return "Frekvens:", max_freq, "D#"
    elif 329.6276 - 5 <= max_freq <= 329.6276 + 5:
        return "Frekvens:", max_freq, "E"
    elif 349.2282 - 5 <= max_freq <= 349.2282 + 5:
        return "Frekvens:", max_freq, "F"
    elif 369.9944 - 5 <= max_freq <= 369.9944 + 5:
        return "Frekvens:", max_freq, "F#"
    elif 391.9954 - 5 <= max_freq <= 391.9954 + 5:
        return "Frekvens:", max_freq, "G"
    elif 415.3047 - 5 <= max_freq <= 415.3047 + 5:
        return "Frekvens:", max_freq, "G#"
    elif 440.0000 - 5 <= max_freq <= 440.0000 + 5:
        return "Frekvens:", max_freq, "A"
    elif 466.1638 - 5 <= max_freq <= 466.1638 + 5:
        return "Frekvens:", max_freq, "A#"
    else:
        return "Frekvens:", max_freq, "B"


audio_files = ['Piano_1_C.wav', 'Piano_2.wav', 'Piano_3.wav', 'Piano_4.wav', 'Piano_5.wav', 'Piano_6.wav']
for audio_file in audio_files:
    audio_file = scipy.io.wavfile.read(audio_file)
    print(get_pitch(audio_file))

