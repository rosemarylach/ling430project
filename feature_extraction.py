# import libraries
import numpy as np
import pandas as pd
import scipy
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt

training_dir = "training/"
training_files = os.listdir(training_dir)
samp_rate = 22050
phonemes = ["t", "th", "tt"]

# create feature dataframe
feature_names = ["mean", "stdev", "skew", "kurtosis", "zcr_mean", "zcr_stdev",
                 "rmse_mean", "rmse_stdev", "tempo"] + \
                ['mfccs_' + str(i+1) + '_mean' for i in range(20)] + \
                ['mfccs_' + str(i+1) + '_stdev' for i in range(20)] + \
                ['chroma_' + str(i+1) + '_mean' for i in range(12)] + \
                ['chroma_' + str(i+1) + '_stdev' for i in range(12)] + \
                ["centroid_mean", "centroid_stdev"] + \
                ['contrast_' + str(i+1) + '_mean' for i in range(7)] + \
                ['contrast_' + str(i+1) + '_std' for i in range(7)] + \
                ["rolloff_mean", "rolloff_stdev", "phoneme"]

param_names = feature_names[1:-1]
label_names = feature_names[-1]

feature_frame = pd.DataFrame(columns=feature_names)

# populate feature dataframe

for wav in training_files:

    # separate phoneme label from filename
    file_info = wav.split("-")
    start_idx = 1 if len(file_info[0]) < 5 else 2
    end_idx = 1 if len(file_info[0]) == 3 else 2 if len(file_info[0]) < 6 else 3
    phoneme = file_info[0][start_idx:end_idx+1]
    label = phonemes.index(phoneme)

    # extract waveform
    y, sr = librosa.load(training_dir + wav, sr = samp_rate)

    # calculate features

    # spectral moments
    mean = np.mean(abs(y))
    stdev = np.std(y)
    skew = scipy.stats.skew(abs(y))
    kurtosis = scipy.stats.kurtosis(y)

    # zero crossing
    zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]
    zcr_mean = np.mean(zcr)
    zcr_stdev = np.std(zcr)

    # root mean squared energy
    rmse = librosa.feature.rms(y + 0.0001)[0]
    rmse_mean = np.mean(rmse)
    rmse_stdev = np.mean(rmse)

    # tempo
    tempo = librosa.beat.tempo(y, sr=sr)

    # Mel-Frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_stdev = np.std(mfccs, axis=1)

    # chroma vector data
    chroma = librosa.feature.chroma_stft(y, sr=sr, hop_length=1024)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_stdev = np.std(chroma, axis=1)

    # spectral centroids
    spectral_centroids = librosa.feature.spectral_centroid(y+0.01, sr=sr)[0]
    centroid_mean = np.mean(spectral_centroids)
    centroid_stdev = np.std(spectral_centroids)

    # spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands = 6, fmin = 200.0)
    contrast_mean = np.mean(spectral_contrast, axis=1)
    contrast_stdev = np.std(spectral_contrast, axis=1)

    # spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr, roll_percent = 0.85)[0]
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_stdev = np.std(spectral_rolloff)
    
    # Build feature vector
    feature_vect = []
    feature_vect.extend([mean, stdev, skew, kurtosis])
    feature_vect.extend([zcr_mean, zcr_stdev])
    feature_vect.extend([rmse_mean, rmse_stdev])
    feature_vect.extend(tempo)
    feature_vect.extend(mfccs_mean)
    feature_vect.extend(mfccs_stdev)
    feature_vect.extend(chroma_mean)
    feature_vect.extend(chroma_stdev)
    feature_vect.extend([centroid_mean, centroid_stdev])
    feature_vect.extend(contrast_mean)
    feature_vect.extend(contrast_stdev)
    feature_vect.extend([rolloff_mean, rolloff_stdev])
    feature_vect.append(phoneme)

    # update dataframe
    feature_frame = feature_frame.append(pd.DataFrame(feature_vect, index=feature_names).transpose(), ignore_index=True)

# save features to csv
feature_frame.to_csv('training_features.csv', index=False)


# %% extract test features

test_dir = "testing/"
test_files = os.listdir(test_dir)

# create feature dataframe
test_names = ["mean", "stdev", "skew", "kurtosis", "zcr_mean", "zcr_stdev",
                "rmse_mean", "rmse_stdev", "tempo"] + \
                ['mfccs_' + str(i+1) + '_mean' for i in range(20)] + \
                ['mfccs_' + str(i+1) + '_stdev' for i in range(20)] + \
                ['chroma_' + str(i+1) + '_mean' for i in range(12)] + \
                ['chroma_' + str(i+1) + '_stdev' for i in range(12)] + \
                ["centroid_mean", "centroid_stdev"] + \
                ['contrast_' + str(i+1) + '_mean' for i in range(7)] + \
                ['contrast_' + str(i+1) + '_std' for i in range(7)] + \
                ["rolloff_mean", "rolloff_stdev", "filename"]

test_frame = pd.DataFrame(columns=test_names)

# populate feature dataframe

for wav in test_files:

    # extract waveform
    y, sr = librosa.load(test_dir + wav, sr = samp_rate)

    # calculate features

    # spectral moments
    mean = np.mean(abs(y))
    stdev = np.std(y)
    skew = scipy.stats.skew(abs(y))
    kurtosis = scipy.stats.kurtosis(y)

    # zero crossing
    zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]
    zcr_mean = np.mean(zcr)
    zcr_stdev = np.std(zcr)

    # root mean squared energy
    rmse = librosa.feature.rms(y + 0.0001)[0]
    rmse_mean = np.mean(rmse)
    rmse_stdev = np.mean(rmse)

    # tempo
    tempo = librosa.beat.tempo(y, sr=sr)

    # Mel-Frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_stdev = np.std(mfccs, axis=1)

    # chroma vector data
    chroma = librosa.feature.chroma_stft(y, sr=sr, hop_length=1024)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_stdev = np.std(chroma, axis=1)

    # spectral centroids
    spectral_centroids = librosa.feature.spectral_centroid(y+0.01, sr=sr)[0]
    centroid_mean = np.mean(spectral_centroids)
    centroid_stdev = np.std(spectral_centroids)

    # spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands = 6, fmin = 200.0)
    contrast_mean = np.mean(spectral_contrast, axis=1)
    contrast_stdev = np.std(spectral_contrast, axis=1)

    # spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr, roll_percent = 0.85)[0]
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_stdev = np.std(spectral_rolloff)
    
    # Build feature vector
    test_vect = []
    test_vect.extend([mean, stdev, skew, kurtosis])
    test_vect.extend([zcr_mean, zcr_stdev])
    test_vect.extend([rmse_mean, rmse_stdev])
    test_vect.extend(tempo)
    test_vect.extend(mfccs_mean)
    test_vect.extend(mfccs_stdev)
    test_vect.extend(chroma_mean)
    test_vect.extend(chroma_stdev)
    test_vect.extend([centroid_mean, centroid_stdev])
    test_vect.extend(contrast_mean)
    test_vect.extend(contrast_stdev)
    test_vect.extend([rolloff_mean, rolloff_stdev])
    test_vect.append(wav)

    # update dataframe
    test_frame = test_frame.append(pd.DataFrame(test_vect, index=test_names).transpose(), ignore_index=True)

# save features to csv
test_frame.to_csv('test_features.csv', index=False)

# %% create mel spectrograms for testing
for wav in test_files:

    # extract waveform
    y, sr = librosa.load(test_dir + wav, sr = samp_rate)

    # create spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, fmax=8000)
    spec_dB = librosa.power_to_db(spec, ref=np.max)

    fig, ax = plt.subplots()

    img = librosa.display.specshow(spec_dB,  sr=sr,
                         fmax=8000, ax=ax)

    #ax.set(title='Mel-frequency spectrogram')
    plt.axis('off')

    plt.savefig('testspecs/' + wav[0:-4] + '.png', bbox_inches='tight', pad_inches=0)

    plt.close()

# %% create mel spectrograms for training
for wav in training_files:

    # extract waveform
    y, sr = librosa.load(training_dir + wav, sr = samp_rate)

    # create spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, fmax=8000)
    spec_dB = librosa.power_to_db(spec, ref=np.max)

    fig, ax = plt.subplots()

    img = librosa.display.specshow(spec_dB,  sr=sr,
                         fmax=8000, ax=ax)

    #ax.set(title='Mel-frequency spectrogram')
    plt.axis('off')

    plt.savefig('trainspecs/' + wav[0:-4] + '.png', bbox_inches='tight', pad_inches=0)

    plt.close()