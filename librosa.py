import librosa
import IPython.display

# Load the audio data
data, sr = librosa.load('path/to/audio/file.wav')

# Write the audio data to an output file
librosa.output.write_wav('output.wav', data, sr)

# Play the audio file using IPython.display
IPython.display.Audio('output.wav')
