import math
import random
import numpy as np
import pandas as pd

def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """
    Counts all n-grams in the given dataset.

    Args:
        data (list of list of str): A list of sentences, where each sentence is a list of words.
        n (int): The n-gram size (e.g., 1 for unigram, 2 for bigram, etc.).
        start_token (str): Special token to indicate the beginning of a sentence.
        end_token (str): Special token to indicate the end of a sentence.
    
    Returns:
        dict: A dictionary mapping n-gram tuples to their frequency count.
    """
    n_grams = {}
    
    # Add start and end tokens to each sentence
    padded_data = [[start_token] * n + sentence + [end_token] for sentence in data]
    
    for sentence in padded_data:
        for i in range(len(sentence) - n + 1):
            n_gram = tuple(sentence[i:i + n])
            n_grams[n_gram] = n_grams.get(n_gram, 0) + 1
    
    return n_grams

def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size):
    """
    Estimates the probability of a word given a previous n-gram using Laplace smoothing.

    Args:
        word (str): The target word to predict.
        previous_n_gram (tuple): The preceding sequence of words (n-gram).
        n_gram_counts (dict): Counts of n-grams.
        n_plus1_gram_counts (dict): Counts of (n+1)-grams.
        vocabulary_size (int): Total number of unique words.

    Returns:
        float: The estimated probability of the given word occurring after the previous n-gram.
    """
    previous_n_gram = tuple(previous_n_gram)
    n_plus1_gram = previous_n_gram + (word,)
    
    n_plus1_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    
    return (n_plus1_count + 1) / (n_gram_count + vocabulary_size)

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary):
    """
    Computes probabilities of all words in the vocabulary appearing after a given n-gram.

    Args:
        previous_n_gram (tuple): The preceding sequence of words (n-gram).
        n_gram_counts (dict): Counts of n-grams.
        n_plus1_gram_counts (dict): Counts of (n+1)-grams.
        vocabulary (list): List of unique words in the dataset.

    Returns:
        dict: A dictionary mapping words to their estimated probability.
    """
    vocabulary_size = len(vocabulary)
    probabilities = {}
    
    for word in vocabulary:
        probabilities[word] = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size)
    
    return probabilities

def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token='<unk>', start_with=None):
    """
    Suggests the most likely next word based on n-gram probabilities.
    
    Args:
        previous_tokens (list): A list of words representing the input sentence.
        n_gram_counts (dict): Dictionary of n-gram frequencies.
        n_plus1_gram_counts (dict): Dictionary of (n+1)-gram frequencies.
        vocabulary (list): List of unique words in the dataset.
        end_token (str): Special token indicating the end of a sentence.
        unknown_token (str): Token for unknown words.
        start_with (str, optional): If provided, filters suggested words by starting substring.
        
    Returns:
        tuple: The most likely next word and its probability.
    """
    n = len(list(n_gram_counts.keys())[0])  # Get n from n-gram dictionary keys
    previous_n_gram = tuple(previous_tokens[-n:])
    
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary)
    
    # Sort words by probability in descending order
    sorted_probabilities = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    
    if start_with:
        sorted_probabilities = [item for item in sorted_probabilities if item[0].startswith(start_with)]
    
    return sorted_probabilities[0] if sorted_probabilities else (unknown_token, 0.0)

if __name__ == "__main__":
    # Example sentences, replace this with whole script of movie to get good result
    sentences = [['i', 'have', 'an', 'army'],
                 ['we', 'have', 'a', 'hulk'],
                 ['the', 'tesseract', 'belongs', 'to', 'me'],
                 ['i', 'am', 'iron', 'man']]
    
    unique_words = list(set(sum(sentences, []))) + ["<s>", "<e>", "<unk>"]
    
    unigram_counts = count_n_grams(sentences, 1)
    bigram_counts = count_n_grams(sentences, 2)
    trigram_counts = count_n_grams(sentences, 3)
    
    previous_tokens = ["i", "am"]
    suggestion, probability = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words)
    
    print(f"{' '.join(previous_tokens)} {suggestion} | probability of {probability:.4f}")
