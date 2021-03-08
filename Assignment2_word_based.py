
import os
import math
import re
import pickle
import time 
import random
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding,Bidirectional,Activation, dot, concatenate,TimeDistributed
from keras.initializers import *
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#set batch size and epoch number
BATCH_SIZE = 64
EPOCH = 3
NUM_WORDS = 5000

#Load the dataset
def load_corpus(filepath):
  with open(filepath,'r',encoding='cp1252') as file:
    dataset = file.read().split('\n')
  return dataset

#this function is to save all used parameters into pickle file
def save_params(filepath, params):
  parameters = {'max_encoder_length': params['max_encoder_length'],
   'max_decoder_length': params['max_decoder_length'],
   'encoder_size': params['encoder_size'],
   'decoder_size': params['decoder_size'],
   'reverse_encoder_token': params['reverse_encoder_token'],
   'reverse_decoder_token': params['reverse_decoder_token'],
   'encoder_token': params['encoder_token'],
   'decoder_token': params['decoder_token']}

  with open(filepath, 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

#This function is to add padding and 
def get_text_encodings_training(texts, parameters):
    enc_seq = parameters["encoder_token"].texts_to_sequences(texts)
    pad_seq = pad_sequences(enc_seq, maxlen=parameters["max_encoder_length"],padding='post')
    pad_seq = to_categorical(pad_seq, num_classes=parameters["encoder_size"])
    return pad_seq

#this function is to add padding for the output and convert it into matrix
def get_text_decodings_training(texts, parameters):
    dec_seq = parameters['decoder_token'].texts_to_sequences(texts)
    pad_seq = pad_sequences(dec_seq, maxlen=parameters['max_decoder_length'], padding='post')
    pad_seq = to_categorical(pad_seq, num_classes=parameters['decoder_size'])
  
    return pad_seq

#This function is to get the matrix of the encoded word
def get_text_encodings(texts, parameters):
  result = []
  for x in texts.split(' '):
    try:
      if parameters["encoder_token"].word_index[x] > parameters["encoder_size"]:
        result.append(parameters["encoder_token"].word_index['<oov>'])
      else:  
        result.append(parameters["encoder_token"].word_index[x])
    except:
      continue
  pad_seq = pad_sequences([result], maxlen=parameters["max_encoder_length"],padding='post')
  pad_seq = to_categorical(pad_seq, num_classes=parameters["encoder_size"])
  return pad_seq

###This model is taken from internet(FastPunct). This function is basically set the model used
def get_model_instance(parameters):
  ###Setting for model, reference from FastPunct
  encoder_inputs = Input(shape=(None, parameters["encoder_size"],))
  encoder = Bidirectional(LSTM(128, return_sequences=True, return_state=True),
                           merge_mode='concat')
  encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

  encoder_h = concatenate([forward_h, backward_h])
  encoder_c = concatenate([forward_c, backward_c])

  decoder_inputs = Input(shape=(None, parameters["decoder_size"],))
  decoder_lstm = LSTM(256, return_sequences=True)
  decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[encoder_h, encoder_c])

  attention = dot([decoder_outputs, encoder_outputs], axes=(2, 2))
  attention = Activation('softmax', name='attention')(attention)
  context = dot([attention, encoder_outputs], axes=(2, 1))
  decoder_combined_context = concatenate([context, decoder_outputs])

  output = TimeDistributed(Dense(128, activation="relu"))(decoder_combined_context)
  output = TimeDistributed(Dense(parameters["decoder_size"], activation="softmax"))(output)

  model = Model([encoder_inputs, decoder_inputs], [output])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

#this function os to generate batch for training
def generate_batch(x, y, parameters, batch_size=64):
  while True:
    for j in range(0, len(x), batch_size):
      #initialize the matrix to store the input and output for encoder and decoder
      encoder_input_data = np.zeros((batch_size,parameters['max_encoder_length'] , parameters['encoder_size']), dtype='float32')

      decoder_input_data = np.zeros((batch_size, parameters['max_decoder_length'], parameters['decoder_size']), dtype='float32')

      decoder_target_data = np.zeros((batch_size, parameters['max_decoder_length'], parameters['decoder_size']), dtype='float32')

      for i, (input_text, target_text) in enumerate(zip(x[j:j+batch_size], y[j:j+batch_size])):
        #convert the text into matrix
        encoder_input_data[i] = get_text_encodings_training([input_text],parameters)
        decoder_input_data[i] = get_text_decodings_training([target_text],parameters)
        decoder_target_data[i] = get_text_decodings_training([target_text[1:]],parameters)


      yield([encoder_input_data, decoder_input_data], decoder_target_data)

#this function is to save the model
def save_model(model,filepath):
  model.save_weights(filepath)

#this function is to load the model
def load_model(model,filepath):
  model.load_weights(filepath)
  return model

#This function is to load the parameters to be used
def load_params(filepath):
  with open(filepath, "rb") as file:
    parameters = pickle.load(file)
  
  return parameters;

#this function is to normalize the input of user
def normalize_input(text):
  result = text.lower()
  txtarr = result.split(' ')
  newtext=""
  for i in range(len(txtarr)):
    txtarr[i] = re.sub(r'\A\W*(\w*)',r'\1', txtarr[i])
    txtarr[i] = re.sub(r'(\w*)\W*\Z',r'\1', txtarr[i])

  newtext = txtarr[0]

  for x in range(len(txtarr)):
    if(x is 0):
      continue
    if newtext[:-1] is not " ":
      newtext = newtext+ " " + txtarr[x]
    else:
      newtext = newtext + txtarr[x]

  newtext =re.sub(r'[ ]{2,}',r' ',newtext)
  return newtext

#This function is to return a list of allowed extra characters
def get_extra_chars(parameters):
  allowed_extras = []
  y = '!"&\'‘“”’+,-.:;…?@^_~'
  for i in y:
    try:
      allowed_extras.append(parameters['decoder_token'].word_index[i])
    except:
      continue;
  return allowed_extras

#this function is to load all needed data for testing
def load_for_test(model_path,param_path):
  param = load_params(param_path)
  model = load_model(get_model_instance(param),model_path)
  return model,param

#this function is to encode input text and predict and decode the output text. It is to return result of the input sentence. Original code from FastPunct
def decode(model, parameters, input_texts, allowed_extras, batch_size):

  parameters["reverse_decoder_token"][0] = "\n"
  outputs = ""
  #convert the input text to be encode
  input_sequences = get_text_encodings(input_texts, parameters)
  target_text = "\t"
  #convert the output text to be input of decoder
  target_seq = parameters["decoder_token"].texts_to_sequences(target_text)
  target_seq = pad_sequences(target_seq, maxlen=parameters["max_decoder_length"],padding="post")
  target_seq_hot = to_categorical(target_seq, num_classes=parameters["decoder_size"])
  extra_char_count = 0
  prev_char_index = 0
  i = 0
  while i<parameters["max_decoder_length"]:
    #get the current translating character
    curr_char_index = i - extra_char_count
    input_encodings = np.argmax(input_sequences, axis=2)
    try:
      if curr_char_index < len(input_texts):
        cur_inp_list = input_encodings[0][curr_char_index]
      else:
        cur_inp_list = 0
    except:
      break
    #get a list of possible output of decoder
    output_tokens = model.predict([input_sequences, target_seq_hot], batch_size=batch_size)
    sampled_possible_indices = np.argsort(output_tokens[:, i, :])[:, ::-1].tolist()
    sampled_token_indices = []
    #store the most possible output
    for index in sampled_possible_indices[0]:
      try:
        if index in allowed_extras:
          if parameters["reverse_decoder_token"][index] == '\n' and cur_inp_list != 0:
            continue
          elif parameters["reverse_decoder_token"][index] != '\n' and prev_char_index in allowed_extras:
            continue
          elif parameters["reverse_decoder_token"][index] == '\t':
            continue
          sampled_token_indices.append(index)
          extra_char_count += 1
          break
        elif parameters["encoder_token"].word_index[parameters["reverse_decoder_token"][index].lower()] == cur_inp_list:
          sampled_token_indices.append(index)
          break
      except:
        continue

    if len(sampled_token_indices) is not 0:
      sampled_chars = parameters["reverse_decoder_token"][sampled_token_indices[0]]
    else:
      
      try:
        sampled_chars = input_texts.split(' ')[curr_char_index].upper()
        x = parameters['decoder_token'].word_index[sampled_chars.lower()]
        if x > parameters["max_decoder_length"]:
          sampled_token_indices.append(parameters['decoder_token'].word_index['<oov>'])
        else:
          sampled_token_indices.append(x)
      except:
        sampled_token_indices.append(parameters['decoder_token'].word_index['<oov>'])

    if sampled_chars=='<oov>':
      sampled_chars = input_texts.split(' ')[curr_char_index].upper()

    #concatenate the predicted result to output
    outputs += " " + sampled_chars
    
    #continue or finish
    if sampled_chars == '\n' or i == parameters["max_decoder_length"] - 1 or curr_char_index is len(input_texts.split(" ")):
      break

    target_seq[:, i + 1] = sampled_token_indices
    target_seq_hot = to_categorical(target_seq, num_classes=parameters["decoder_size"])
    prev_char_index = sampled_token_indices[0]
    i += 1

  return outputs.strip()

#this function is to return the result of the predicted output
def getResult(text,model,param):
  return decode(model, param,normalize_input(text), get_extra_chars(param), BATCH_SIZE)

#function to start training the model
def training():
  #Load training and testing set
  train_corpus = load_corpus('2008.txt')
  test_corpus_input = load_corpus('norm.txt')
  test_corpus_output_temp = load_corpus('unnorm.txt')

  test_corpus_output = []
  #preprosessing the test output
  for i in test_corpus_output_temp:
    t = "\t " + i.strip() + " \n"
    txtarr = t.split(' ')
    result = []
    newtext=""
    for k in range(len(txtarr)):
      if k is 0 or k is len(txtarr)-1:
        result.append(txtarr[k])
        continue
      txtarr[k] = re.sub(r'\A(\W*)(\w*)',r'\1 \2', txtarr[k])
      txtarr[k] = re.sub(r'(\w*)(\W*)\Z',r'\1 \2', txtarr[k])
      txtarr[k] = re.sub(r'[ ]{2,}',r' ',txtarr[k])
      for j in range(len(txtarr[k].split(' '))):
        if len(re.findall(r'\A\w.*\w\Z' , txtarr[k].split(' ')[j])) is not 0:
          result.append(txtarr[k].split(' ')[j])
        else:
          z = re.sub(r'([!"#$%&\'‘“”’()*+,-./:;<=>…?@[\]^_`{|}~\n\t])',r' \1 ' ,txtarr[k].split(' ')[j])
          for a in z:
            result.append(a)

    final = ""
    for x in result:
      final+= " " + x
    final = re.sub(r'[ ]{2,}',r' ',final)
    test_corpus_output.append(final)

  #Initialize a list of normalized text for output of training and list to stored characters for input and output 
  train_corpus_output = []
  train_corpus_input = []
  input_word = []
  output_word = []

  for i in train_corpus:
    target_text = '\t '+ i +' \n'  
    txtarr = target_text.split(' ')
    result = []
    newtext=""
    for k in range(len(txtarr)):
      if k is 0 or k is len(txtarr)-1:
        result.append(txtarr[k])
        continue
      txtarr[k] = re.sub(r'\A(\W*)(\w*)',r'\1 \2', txtarr[k])
      txtarr[k] = re.sub(r'(\w*)(\W*)\Z',r'\1 \2', txtarr[k])
      txtarr[k] = re.sub(r'[ ]{2,}',r' ',txtarr[k])
      for j in range(len(txtarr[k].split(' '))):
        if len(re.findall(r'\A\w.*\w\Z' , txtarr[k].split(' ')[j])) is not 0:
          result.append(txtarr[k].split(' ')[j])
        else:
          z = re.sub(r'([!"#$%&\'‘“”’()*+,-./:;<=>…?@[\]^_`{|}~\n\t])',r' \1 ' ,txtarr[k].split(' ')[j])
          for a in z:
            result.append(a)

    final = ""
    for x in result:
      final+= " " + x
    final = re.sub(r'[ ]{2,}',r' ',final)

    train_corpus_output.append(final)

  #Normalization of training set
  for sentence in train_corpus_output:

    #Normalize the sentence into lower case
    training_input = sentence.lower()

    #Remove punctuation of sentence
    txtarr = training_input.split(' ')
    newtext=""
    for i in range(len(txtarr)):
      txtarr[i] = re.sub(r'\A\W*(\w*)',r'\1', txtarr[i])
      txtarr[i] = re.sub(r'(\w*)\W*\Z',r'\1', txtarr[i])

    newtext = txtarr[0]

    for x in range(len(txtarr)):
      if(x is 0):
        continue
      if newtext[:-1] is not " ":
        newtext = newtext+ " " + txtarr[x]
      else:
        newtext = newtext + txtarr[x]

    newtext =re.sub(r'[ ]{2,}',r' ',newtext)
    #newtext
    train_corpus_input.append(newtext.strip())

  #Calculate the longest length for input and output sentence 
  input_longest_sentence = 0
  output_longest_sentence = 0
  for x in range(len(train_corpus_input)):
    if len(train_corpus_input[x].split(' ')) > input_longest_sentence:
      input_longest_sentence = len(train_corpus_input[x].split(' '))
    if len(train_corpus_output[x].split(' ')) > output_longest_sentence:
      output_longest_sentence = len(train_corpus_output[x].split(' '))

  #Tokenization for input and output
  encoder_token = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False, num_words=NUM_WORDS, oov_token='<oov>')
  encoder_token.fit_on_texts(train_corpus_input)
  decoder_token = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False, num_words=NUM_WORDS,oov_token='<oov>')
  decoder_token.fit_on_texts(train_corpus_output)

  #get reverse the word index and word
  reverse_encoder_token = {i: c for c, i in encoder_token.word_index.items()}
  reverse_decoder_token = {i: c for c, i in decoder_token.word_index.items()}

  #set the values of parameters
  parameters = {'max_encoder_length':input_longest_sentence, 'max_decoder_length':output_longest_sentence, 'encoder_size':NUM_WORDS+1 ,'decoder_size':NUM_WORDS+1 , 'reverse_encoder_token':reverse_encoder_token , 'reverse_decoder_token':reverse_decoder_token , 'encoder_token':encoder_token , 'decoder_token':decoder_token  }

  #call the save_params function to save the parameters
  save_params('word_parameters.pkl',parameters)

  #initialize the model to train
  model  = get_model_instance(parameters)

  #start the training for model
  history = model.fit_generator(
    generator=generate_batch(x=train_corpus_input, y=train_corpus_output, parameters=parameters,batch_size=BATCH_SIZE),
    steps_per_epoch=math.ceil(len(train_corpus_input)/BATCH_SIZE),
    epochs=EPOCH,
    verbose=1,
    validation_data = generate_batch(x=test_corpus_input, y=test_corpus_output,parameters = parameters, batch_size=BATCH_SIZE),
    validation_steps=math.ceil(len(test_corpus_input)/BATCH_SIZE),
    workers=1,
  )

  #save the model
  save_model(model,'word_model.h5')

#function to accept input for prediction and output the result
def testing():
  #load model and parameters for testing
  model,param = load_for_test('word_model.h5','word_parameters.pkl') # change to your path

  while True:
    choice = input("Select your choice: \n1. Using training set (Random 10)\n2. Using test set (Random 10)\n*. Enter your own sentence\n")
    if choice is "1":

      #Load training and testing set
      train_corpus = load_corpus('2008.txt')
      
      startFrom = random.randint(0,len(train_corpus)-11)

      #Initialize a list of normalized text for output of training and list to stored characters for input and output 
      train_corpus_output = []
      train_corpus_input = []
      input_word = []
      output_word = []

      train_corpus = train_corpus[startFrom:startFrom+10]

      for i in train_corpus:
        target_text = '\t '+ i +' \n'  
        txtarr = target_text.split(' ')
        result = []
        newtext=""
        for k in range(len(txtarr)):
          if k is 0 or k is len(txtarr)-1:
            result.append(txtarr[k])
            continue
          txtarr[k] = re.sub(r'\A(\W*)(\w*)',r'\1 \2', txtarr[k])
          txtarr[k] = re.sub(r'(\w*)(\W*)\Z',r'\1 \2', txtarr[k])
          txtarr[k] = re.sub(r'[ ]{2,}',r' ',txtarr[k])
          for j in range(len(txtarr[k].split(' '))):
            if len(re.findall(r'\A\w.*\w\Z' , txtarr[k].split(' ')[j])) is not 0:
              result.append(txtarr[k].split(' ')[j])
            else:
              z = re.sub(r'([!"#$%&\'‘“”’()*+,-./:;<=>…?@[\]^_`{|}~\n\t])',r' \1 ' ,txtarr[k].split(' ')[j])
              for a in z:
                result.append(a)

        final = ""
        for x in result:
          final+= " " + x
        final = re.sub(r'[ ]{2,}',r' ',final)

        train_corpus_output.append(final)

      #Normalization of training set
      for sentence in train_corpus_output:

        #Normalize the sentence into lower case
        training_input = sentence.lower()

        #Remove punctuation of sentence
        txtarr = training_input.split(' ')
        newtext=""
        for i in range(len(txtarr)):
          txtarr[i] = re.sub(r'\A\W*(\w*)',r'\1', txtarr[i])
          txtarr[i] = re.sub(r'(\w*)\W*\Z',r'\1', txtarr[i])

        newtext = txtarr[0]

        for x in range(len(txtarr)):
          if(x is 0):
            continue
          if newtext[:-1] is not " ":
            newtext = newtext+ " " + txtarr[x]
          else:
            newtext = newtext + txtarr[x]

        newtext =re.sub(r'[ ]{2,}',r' ',newtext)
        #newtext
        train_corpus_input.append(newtext.strip())

      
      for x in range(len(train_corpus_input)):
        print("Input sentence : ",train_corpus_input[x].strip())
        print("Predicted sentence : ",getResult(train_corpus_input[x],model,param).strip())
        print("Expected sentence : ",train_corpus_output[x].strip())
        print("\n\n")

    elif choice is "2":
      test_corpus_input = load_corpus('norm.txt')
      test_corpus_output = load_corpus('unnorm.txt')
      startFrom = random.randint(0,len(test_corpus_input)-11)
      for x in range(startFrom,startFrom+10):
        print("Input sentence : ",test_corpus_input[x].strip())
        print("Predicted sentence : ",getResult(test_corpus_input[x],model,param).strip())
        print("Expected sentence : ",test_corpus_output[x].strip())
        print("\n\n")
    else:
      sentence = input("Enter sentence to test :    ")
      print("Predicted result : ",getResult(sentence,model,param))
    print("\n\n")
    cont = input("Do you want to continue? Y or N : ")
    if cont is "Y" or cont is "y":
      continue
    elif cont is "N" or cont is "n":
      break

print("Assignment 2 : Restoration of normalized sentence")
while True:
  choice = input("Enter your choice\n1. Training the model \n2. Start testing\n")
  if choice is "1":
    print("\nPlease wait for awhile\n")
    training()
  elif choice is "2":
    testing()
  else:
    print("Invalid input!!")
  
  again = input("Do you want to continue? Y or N : ")
  if again is "Y" or again is "y":
    continue
  elif again is "N" or again is "n":
    break