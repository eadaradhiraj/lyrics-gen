from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import os
import re
import logging

DIR_PATH = os.path.dirname(__file__)
SAVED_MODEL = os.path.join(DIR_PATH, 'model.h5')

chars = [' ',
         "'",
         'a',
         'b',
         'c',
         'd',
         'e',
         'f',
         'g',
         'h',
         'i',
         'j',
         'k',
         'l',
         'm',
         'n',
         'o',
         'p',
         'q',
         'r',
         's',
         't',
         'u',
         'v',
         'w',
         'x',
         'y',
         'z']

char_indices = {
    ' ': 0,
    "'": 1,
    'a': 2,
    'b': 3,
    'c': 4,
    'd': 5,
    'e': 6,
    'f': 7,
    'g': 8,
    'h': 9,
    'i': 10,
    'j': 11,
    'k': 12,
    'l': 13,
    'm': 14,
    'n': 15,
    'o': 16,
    'p': 17,
    'q': 18,
    'r': 19,
    's': 20,
    't': 21,
    'u': 22,
    'v': 23,
    'w': 24,
    'x': 25,
    'y': 26,
    'z': 27
}

indices_char = {
    0: ' ',
    1: "'",
    2: 'a',
    3: 'b',
    4: 'c',
    5: 'd',
    6: 'e',
    7: 'f',
    8: 'g',
    9: 'h',
    10: 'i',
    11: 'j',
    12: 'k',
    13: 'l',
    14: 'm',
    15: 'n',
    16: 'o',
    17: 'p',
    18: 'q',
    19: 'r',
    20: 's',
    21: 't',
    22: 'u',
    23: 'v',
    24: 'w',
    25: 'x',
    26: 'y',
    27: 'z'
}


maxlen = 40
step = 3


def generate_lyrics(usr_input):
    sentences = []
    next_chars = []

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def on_epoch_end(epoch, logs):
        logging.info(f'----- Generating text after Epoch: {epoch}')

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5]:
            logging.info(f'----- diversity: {diversity}')

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            logging.info(f'----- Generating with seed: " {sentence} "')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            logging.info('\n')

    if (not os.path.exists(SAVED_MODEL)):
        with open('corpus.txt', 'r') as inp:
            cleaned = str(inp.read()).lower().replace(' ', '\n')
            text = " ".join(re.findall(r"[a-z']+", cleaned))

        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])

        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.01))

        model.summary()

        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

        model.fit(
            x,
            y,
            batch_size=128,
            epochs=2,
            callbacks=[print_callback]
        )

        model.save(SAVED_MODEL)

    else:
        model = load_model(SAVED_MODEL)

    generated = ''
    Tx = 40
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input

    for i in range(400):

        x_pred = np.zeros((1, Tx, len(chars)))

        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature=0.2)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        if next_char == '\n':
            continue
    return generated


print(generate_lyrics("Shook up"))
