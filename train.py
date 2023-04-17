import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('data/datasetaceathar_randomised_2.csv', sep=";", encoding='utf8')

# Split the data into questions and answers
questions = data['Question'].values
answers = data['Answer'].values

texts = pd.concat([data['Question'], data['Answer']], axis=0).astype("str")

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

# Save tokenizer
with open('tokens/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert text to sequences
question_sequences = tokenizer.texts_to_sequences(questions.astype("str"))
answer_sequences = tokenizer.texts_to_sequences(answers.astype("str"))

# Pad the sequences
max_length = max([len(seq) for seq in question_sequences])
question_sequences = tf.keras.preprocessing.sequence.pad_sequences(question_sequences, maxlen=max_length, padding='post')
answer_sequences = tf.keras.preprocessing.sequence.pad_sequences(answer_sequences, maxlen=max_length, padding='post')

# Split the data into training and testing sets
question_train, question_test, answer_train, answer_test = train_test_split(
    question_sequences, answer_sequences, test_size=0.1, random_state=42)

##original seq2seq
# # Define the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
#     #tf.keras.layers.LSTM(128, return_sequences=True),
#     tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
# ])

##attention layer
# Define the model
# input_layer = tf.keras.layers.Input(shape=(max_length,))
# embedding_layer = tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length)(input_layer)
# bidirectional_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(embedding_layer)
#
# # Attention layer
# attention_layer = tf.keras.layers.Attention()([bidirectional_lstm, bidirectional_lstm])
#
# # Concatenate the attention output with the bidirectional LSTM output
# concat_layer = tf.keras.layers.Concatenate()([bidirectional_lstm, attention_layer])
#
# # TimeDistributed layer
# output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))(concat_layer)
#
# # Create the model
# model = tf.keras.Model(inputs=input_layer, outputs=output_layer)


# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dynamic learning rate schedule
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# stops the training when validation loss does not improve for 10 consecutive epochs
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(question_train, answer_train, epochs=200, validation_data=(question_test, answer_test), callbacks=[reduce_lr])

# Save the model
model.save('models/model_200_4_randomised_2_attention17.h5')

# Evaluate the model on the test set
loss, accuracy = model.evaluate(question_test, answer_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
print('Epochs', 200)
