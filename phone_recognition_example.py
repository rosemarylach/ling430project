# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 23:28:23 2022

@author: jtman
"""


import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 2  # Duration of recording
import librosa

pre_word_list = ["sina", "ta", "titi", "suta" , "nuna", "tanatana", "sisu", "tutana", "tisini", "nutuna", "nitusu", "tanusa", "sutusa","nanata", "nisa", "tusinani", "tasu", "susa", "nasu", "nutuna", "si", "sasuna", "titu", "tana", "satinasa", "tisinu", "nusi"]
vocab = set([char for word in pre_word_list for char in word] + ['\t', '\n'])

word_list = []
for word in pre_word_list:
    new_word = '\t' + word + '\n'
    word_list.append(new_word)
    

label_char_dict = {char : i for i, char in enumerate(vocab)}
label_num_dict = {i: char for i, char in enumerate(vocab)}

# for word in word_list:
#     print ("Recording... " + str(word))
#     myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
#     sd.wait()  # Wait until recording is finished
#     tempwordwav = "D:/speech_recognition/" + word + ".wav"
#     write(tempwordwav, fs, myrecording) 


# mfccs = []
# num_mfccs = 12    
# max_out_length = 10
# labels_in = []
# labels_out = []

# import numpy as np
# decoder_input = np.zeros((len(word_list), max_out_length, len(vocab)), dtype = 'float32')
# decoder_output = np.zeros((len(word_list), max_out_length, len(vocab)), dtype = 'float32')

# for i, item in enumerate(word_list):
#     y1, sr1 = librosa.load("D:/speech_recognition/" + item[1:-1] + ".wav")
#     mfccs.append(librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=num_mfccs))
    
#     for pos, char in enumerate(item):
#         decoder_input[i, pos, label_char_dict[char]] = 1
#         if pos > 0:
#             decoder_output[i, pos - 1, label_char_dict[char]] = 1
 

# mfccs = np.array(mfccs)
    
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense

# batch_size = 1
# epochs = 1000
# num_neurons = 128

# encoder_inputs = Input(shape = (num_mfccs, 87))
# encoder = LSTM(num_neurons, return_state = True)
# encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# encoder_states = [state_h, state_c]   #final states and memory that we input into decoder LSTM

# decoder_inputs = Input(shape = (None, len(vocab)))
# decoder_lstm = LSTM (num_neurons, return_sequences = True, return_state = True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)  #setting initial state as final encoder state
# decoder_dense = Dense(len(vocab), activation = 'softmax')
# decoder_outputs = decoder_dense(decoder_outputs)
# #decoder_dense = Dense(len(vocab), activation = 'softmax')
# #decoder_outputs = decoder_dense(decoder_outputs1)
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

# model.fit([mfccs, decoder_input], decoder_output, batch_size = batch_size, epochs = epochs) #, validation_split=0.1)


# model_structure = model.to_json()
# with open("phonetic_model7.json", "w") as json_file:
#     json_file.write(model_structure)

# model.save_weights("phonetic_model7.h5")
    








from keras.models import model_from_json
with open("phonetic_model6.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)
model.load_weights('phonetic_model6.h5')


pre_word_list = ["sina", "ta", "titi", "suta" , "nuna", "tanatana", "sisu", "tutana", "tisini", "nutuna", "nitusu", "tanusa", "sutusa","nanata", "nisa", "tusinani", "tasu", "susa", "nasu", "nutuna", "si", "sasuna", "titu", "tana", "satinasa", "tisinu", "nusi"]
vocab = set([char for word in pre_word_list for char in word] + ['\t', '\n'])

word_list = []
for word in pre_word_list:
    new_word = '\t' + word + '\n'
    word_list.append(new_word)
    

label_char_dict = {char : i for i, char in enumerate(vocab)}
label_num_dict = {i: char for i, char in enumerate(vocab)}



import numpy as np

from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

num_mfccs = 12
num_neurons = 256

encoder_inputs = model.input[0]
encoder_outputs, state_h, state_c = model.layers[2].output
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]
decoder_state_input_h = Input(shape=(num_neurons,))
decoder_state_input_c = Input(shape=(num_neurons,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
#decoder_dense = model.layers[5]
#decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


stop_token = '\n'
for x in range(20):
    print ("Recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    tempwordwav = "D:/speech_recognition/predict" + ".wav"
    write(tempwordwav, fs, myrecording)
    
    ynew, srnew = librosa.load("D:/speech_recognition/predict.wav")
    new_mfccs = []
    new_mfccs.append(librosa.feature.mfcc(y=ynew, sr=srnew, n_mfcc=num_mfccs))
    new_mfccs = np.array(new_mfccs)
    
    
    thought = encoder_model.predict(new_mfccs)
    target_seq = np.zeros((1, 1, len(vocab)))
    stop_condition = False
    generated_sequence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + thought)
        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_word = label_num_dict[generated_token_idx]
        generated_sequence += generated_word + " "
        if (generated_word == stop_token):
            stop_condition = True
        target_seq = np.zeros((1, 1, len(vocab)))
        target_seq[0, 0, generated_token_idx] = 1.
        thought = [h, c]
    
    print (generated_sequence)