import librosa

# Load the audio signal and sample rate
signal, sr = librosa.load('path/to/audio.wav')

# Extract the MFCCs using the default settings
mfccs = librosa.feature.mfcc(signal, sr)

# Print the shape of the MFCCs array
print(mfccs.shape)
#The mfccs variable will contain a 2D array of shape (num_mfccs, num_frames), 
#where num_mfccs is the number of MFCCs that were extracted (by default, this is 20), and num_frames is the number of frames in the audio signal.



# Extract the MFCCs with custom settings
mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=2048, hop_length=512)

#In this example, the number of MFCCs is increased to 40, the FFT window size is increased to 2048 samples, and the hop length is decreased to 512 samples.
#You can experiment with these settings to find the values that work best for your specific audio signal and application.
