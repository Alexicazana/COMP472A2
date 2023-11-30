import gensim
import gensim.downloader as api
import pandas as pd
import numpy as np
import os
from gensim.models import KeyedVectors

# Function to find the closest synonym
def find_closest_synonym(row, model):
    question_word = row['question']
    answer_word = row['answer']
    guess_words = [row[str(i)] for i in range(4)]

    # Check if question word is in model
    if question_word not in model.key_to_index:
        return question_word, answer_word, None, 'guess'

    # Filter out guess words not in the model
    valid_guess_words = [word for word in guess_words if word in model.key_to_index]

    # If no valid guess words, return guess label
    if not valid_guess_words:
        return question_word, answer_word, None, 'guess'

    # Compute similarities and find guess word with highest similarity
    similarities = [model.similarity(question_word, guess_word) for guess_word in valid_guess_words]
    best_guess = valid_guess_words[np.argmax(similarities)]

    # Determine label according to reqs
    label = 'correct' if best_guess == answer_word else 'wrong'

    return question_word, answer_word, best_guess, label

# 1. Glove and fast text => different corpora, same embedding size
# 2. glove-twitter 25, 50 => same copora, different embedding size
model_list = ["glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300", "glove-twitter-50", "glove-twitter-25"]

# load synonym data set
synonym_data = pd.read_csv('synonym.csv', dtype=str)

# to output to task 2 directory, do not want to confuse the files
output_dir = '../task2'
os.makedirs(output_dir, exist_ok=True)

# set analysis.csv output path
analysis_file_path = os.path.join(output_dir, 'analysis.csv')

# Same as task1 but we have multiple models now so lets iterate over the models
for m in model_list:
    print(f"Now Processing Model: {m}")
    model = api.load(m)

    # Apply function to each row in dataset
    synonym_data['results'] = synonym_data.apply(lambda row: find_closest_synonym(row, model), axis=1)
    # Extract results
    synonym_data[['question-word', 'answer-word', 'guess-word', 'label']] = pd.DataFrame(
        synonym_data['results'].tolist(), index=synonym_data.index)

    # Drop original columns and results column
    synonym_data_cleaned = synonym_data.drop(columns=['question', 'answer', '0', '1', '2', '3', 'results'])

    # save as csv file
    details_file_name = f"{m.replace('-', '_')}-details.csv"
    details_file_path = os.path.join(output_dir, details_file_name)
    synonym_data_cleaned.to_csv(details_file_path, index=False)

    # analysis data
    vocab_size = len(model.key_to_index)
    correct_count = synonym_data_cleaned['label'].value_counts().get('correct', 0)
    non_guess_questions = len(synonym_data_cleaned) - synonym_data_cleaned['label'].value_counts().get('guess', 0)
    accuracy = correct_count / non_guess_questions if non_guess_questions > 0 else 0

    # analysis dataframe for output
    analysis_data = pd.DataFrame({
        "model_name": [m],
        "vocab_size": [vocab_size],
        "correct_count": [correct_count],
        "non_guess_questions": [non_guess_questions],
        "accuracy": [accuracy]
    })

    # create the file if does not exists, else append analysis data
    if not os.path.isfile(analysis_file_path):
        analysis_data.to_csv(analysis_file_path, index=False)
    else:
        analysis_data.to_csv(analysis_file_path, mode='a', header=False, index=False)

    print(f"Finished processing {m}. Details: {details_file_path}, Analysis appended.")
