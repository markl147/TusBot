import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

# Save the users question, the incorrect response as designated by the user, and the users feedback, adds them to a list and when the program ends exports them to a csv file
def export_feedback_to_csv(user_question, bad_bot_response, user_response):
    data_dict = {'question': user_question_list, 'response': bot_response_list, 'feedback': user_answer_list}
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv('data/user_feedback.csv', index=False)

# Initialize lists to store questions, responses, and feedback
user_question_list = []
bot_response_list = []
user_answer_list = []

# Load the tokenizer
with open('tokens/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = tf.keras.models.load_model('models/model_1000_4_randomised.h5')
model.summary()

# Get max_length from the model
max_length = model.layers[0].input_shape[1]

def user_update_model(question, answer, epochs=5):
    question_seq = tokenizer.texts_to_sequences([question])[0]
    question_seq = tf.keras.preprocessing.sequence.pad_sequences([question_seq], maxlen=max_length, padding='post')
    answer_seq = tokenizer.texts_to_sequences([answer])[0]
    answer_seq = tf.keras.preprocessing.sequence.pad_sequences([answer_seq], maxlen=max_length, padding='post')
    model.fit(question_seq, np.expand_dims(answer_seq, axis=-1), epochs=epochs)

# Define the function to generate the response
def generate_response(question, min_confidence=0.2):
    question_seq = tokenizer.texts_to_sequences([question])[0]
    question_seq = tf.keras.preprocessing.sequence.pad_sequences([question_seq], maxlen=max_length, padding='post')
    prediction = model.predict(question_seq)[0]
    index = np.argmax(prediction, axis=-1)
    confidence = np.mean([prediction[i, idx] for i, idx in enumerate(index) if idx > 0])
    response = ' '.join([tokenizer.index_word[i] for i in index if i > 0])
    print(confidence)

    # Check if the confidence is above the threshold
    if confidence > min_confidence:
        return response
    else:
        return "I'm sorry, I don't have an answer for that."


# return response
while True:
    question = input('You: ')
    if question == 'exit':
        break
    response = generate_response(question)
    print('Bot:', response)

    # Ask for user feedback
    # feedback = input('Is this response satisfactory? (y/n): ')
    # if feedback == 'n':
    #     proper_answer = input('Please provide the correct response: ')

        # #calls the method to update model with correct answer for question
        # user_update_model(question, proper_answer)

        # Store the question, response, and feedback
        # user_question_list.append(question)
        # bot_response_list.append(response)
        # user_answer_list.append(proper_answer)

# trains the model on questions that users provided the correct responses for
# model.save('models/model_user_updated.h5')
# print("model updated")

# does what it says on the tin
# export_feedback_to_csv(user_question_list, bot_response_list, user_answer_list)
# print('Feedback received')
