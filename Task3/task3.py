import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import pandas as pd
import numpy as np
from typing import List
from gensim.models import Word2Vec

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

def preprocess_books(books_directory: str, book_filenames: List[str]) -> List[List[str]]:
    """
    Preprocesses a book file by tokenizing its content into sentences and words.

    Args:
        books_directory: The directory containing the book files.
        book_filenames: A list of filenames of the book files.

    Returns:
        A list of tokenized sentences, where each sentence is a list of words.
    """
    all_sentences = []

    # Iterate over each book filename
    for filename in book_filenames:
        # Open the book file
        with open(os.path.join(books_directory, filename), 'r', encoding='utf-8') as file:
            # Read the content of the file
            content = file.read()

            # Tokenize the text into sentences
            sentences = sent_tokenize(content)

            # Tokenize each sentence into words
            tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

            # Append the tokenized sentences to the list of all sentences
            all_sentences.extend(tokenized_sentences)

    return all_sentences

def train_and_evaluate_models(all_sentences, model_params, synonym_data, output_dir, analysis_file_path):
    """
    Trains and evaluates Word2Vec models using the given parameters.

    Args:
        all_sentences (list): List of sentences to train the models.
        model_params (list): List of tuples containing the size and window parameters for the models.
        synonym_data (pandas.DataFrame): DataFrame containing the synonym test data.
        output_dir (str): Directory to save the model details.
        analysis_file_path (str): File path to save the analysis data.

    Returns:
        None
    """
    for size, window in model_params:
        # Train Word2Vec model
        model = Word2Vec(sentences=all_sentences, vector_size=size, window=window, min_count=1)
        model_name = f"word2vec_size{size}_window{window}"
        print(f"Training model: {model_name}")

        # Evaluate model using synonym test
        synonym_data['results'] = synonym_data.apply(lambda row: find_closest_synonym(row, model.wv), axis=1)
        synonym_data[['question-word', 'answer-word', 'guess-word', 'label']] = pd.DataFrame(synonym_data['results'].tolist(), index=synonym_data.index)
        synonym_data_cleaned = synonym_data.drop(columns=['question', 'answer', '0', '1', '2', '3', 'results'])

        # Save results
        details_file_path = os.path.join(output_dir, f"{model_name}-details.csv")
        synonym_data_cleaned.to_csv(details_file_path, index=False)

        # Calculate analysis data
        vocab_size = len(model.wv.key_to_index)
        correct_count = synonym_data_cleaned['label'].value_counts().get('correct', 0)
        non_guess_questions = len(synonym_data_cleaned) - synonym_data_cleaned['label'].value_counts().get('guess', 0)
        accuracy = correct_count / non_guess_questions if non_guess_questions > 0 else 0

        # Append analysis data to analysis_file_path
        analysis_data = pd.DataFrame({
            "model_name": [model_name],
            "vocab_size": [vocab_size],
            "correct_count": [correct_count],
            "non_guess_questions": [non_guess_questions],
            "accuracy": [accuracy]
        })
        if not os.path.isfile(analysis_file_path):
            analysis_data.to_csv(analysis_file_path, index=False)
        else:
            analysis_data.to_csv(analysis_file_path, mode='a', header=False, index=False)

        print(f"Finished training and evaluating {model_name}. Details saved.")
def main():
    """
    This function is the entry point of the program.
    It downloads the necessary NLTK data, preprocesses a set of books,
    loads synonym data, and trains and evaluates models.

    Args:
        None

    Returns:
        None
    """
    # Download NLTK data
    nltk.download('punkt')

    # Define the directory and filenames of the books
    books_directory = './Books'
    book_filenames = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt']

    # Preprocess the books and get all sentences
    all_sentences = preprocess_books(books_directory, book_filenames)

    # Define the output directory and analysis file path
    output_dir = '.'  # Current directory
    analysis_file_path = '../analysis.csv'

    # Define model parameters (embedding size, window size)
    model_params = [(100, 5), (100, 10), (300, 5), (300, 10)]

    # Load synonym data set
    synonym_data = pd.read_csv('synonym.csv', dtype=str)

    # Train and evaluate models
    train_and_evaluate_models(all_sentences, model_params, synonym_data, output_dir, analysis_file_path)

if __name__ == "__main__":
    main()
