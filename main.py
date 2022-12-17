import string

import pandas as pd
import numpy as np
import tensorflow as tf

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential


# Create a function to build the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(3)
  ])
  return model


if __name__ == '__main__':
    database = pd.read_csv("rsos_dream_data.csv")
    df = database[["text_dream", "Male", "Animal", "Friends", "Family", "Dead&Imaginary",
                                "Aggression/Friendliness", "A/CIndex", "F/CIndex", "S/CIndex", "NegativeEmotions"]]

    df['text_dream'] = df['text_dream'].apply(lambda x: x.lower())
    df['text_dream'] = df['text_dream'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['text_dream'] = df['text_dream'].apply(lambda x: x.translate(str.maketrans('', '', string.digits)))
    df['text_dream'] = df['text_dream'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # Vectorize the text data
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['text_dream'])
    sequences = tokenizer.texts_to_sequences(df['text_dream'])
    X = pad_sequences(sequences, maxlen=100)
    """
    scaler = StandardScaler()
    X_other = scaler.fit_transform(df.drop(columns=['text_dream']))

    # Concatenate the vectorized text data with the other numerical data
    X = np.concatenate((X, X_other), axis=1)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, df[
        ['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'Aggression/Friendliness', 'A/CIndex', 'F/CIndex',
         'S/CIndex', 'NegativeEmotions']], test_size=0.2)
    print(X.shape)
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=100))
    model.add(LSTM(128))
    model.add(Dense(10, activation='relu'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse'])
    model.fit(X_train, y_train, epochs=20, verbose=1, validation_data=(X_test, y_test),
              callbacks=[EarlyStopping(monitor='val_loss', patience=20)])

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

    prediction = model.predict(X_test)
    print(prediction)


