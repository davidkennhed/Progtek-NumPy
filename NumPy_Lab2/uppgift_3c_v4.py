import numpy as np
import scipy
import matplotlib.pyplot as plt

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

def get_note(freq):
    # intervall = 26.16
    # 13.08
    freq = freq*2 if freq < 440 else freq
    notes = {
            'A': (426.92, 453.08),
            'A#': (453.08, 480.8),
            'B': (480.8, 509.08),
            'C': (509.08, 541.29),
            'C#': (541.29, 574.61),
            'D': (574.61, 609.25),
            'D#': (609.25, 645.29),
            'E': (645.29, 682.69),
            'F': (682.69, 721.54),
            'F#': (721.54, 761.83),
            'G': (761.83, 803.61),
            'G#': (803.61, 846.96),
        }

    for note, (f_min, f_max) in notes.items():
        if f_min <= freq < f_max:
            return note

    return 'Frequency out of range'


def overtones(data_audio_file):
    e_overtones = []
    freq_range = (300, 360)
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
        new_tone = tone * 0.94385432473
        print(f"Major tone: {tone}, Minor tone: {new_tone}")
        new_tones_list.append(new_tone)
    return new_tones_list


def freq_to_index(freq):
    index = (freq*88015/44100)
    return index


def index_to_freq(index):
    freq = (index*44100/88015)
    return freq


def C():
    rate_upg_c, data_upg_c = scipy.io.wavfile.read('Cdur.wav')
    frequency_data = scipy.fftpack.fft(data_upg_c)
    length_data_c = len(data_upg_c)
    data_frequency = scipy.fftpack.fftfreq(int(length_data_c), 1 / rate_upg_c)
    plt.plot(data_frequency, abs(frequency_data), label='major')

    indexes_of_e = []
    for i in range(len(frequency_data[0:int(length_data_c / 2)])):
        if abs(frequency_data[i]) > 20000:
            if get_note(data_frequency[i]) == "E":
                indexes_of_e.append(i)

    indexes_of_c_minor = []
    for i in indexes_of_e:
        indexes_of_c_minor.append(int(i * 0.94385432473))

    counter = 0
    for i in indexes_of_c_minor:
        frequency_data[i] = frequency_data[indexes_of_e[counter]]
        frequency_data[-i] = frequency_data[-indexes_of_e[counter]]
        frequency_data[indexes_of_e[counter]], frequency_data[-indexes_of_e[counter]] = 0, 0

        counter += 1

    audio = scipy.fftpack.ifft(frequency_data)
    scipy.io.wavfile.write('Cdur_minor.wav', rate_upg_c, audio.astype(np.int16))

    plt.plot(data_frequency, abs(frequency_data), label='minor')
    plt.legend()
    plt.xlim(0, 1500)
    plt.show()


C()
