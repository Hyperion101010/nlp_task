import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('punkt')
nltk.download('wordnet')

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    
    text = example["text"]
    
    qwerty_map = {
        'a': ['q', 'w', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 'r', 'd', 's'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'],
        'z': ['a', 's', 'x']
    }
    
    tokens = word_tokenize(text)
    transformed_tokens = []
    
    for token in tokens:
        if not token.isalpha():
            transformed_tokens.append(token)
            continue
        
        if random.random() < 0.10:
            synsets = wordnet.synsets(token.lower())
            if synsets:
                synonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if ' ' not in synonym and synonym.lower() != token.lower() and synonym.isalpha():
                            synonyms.append(synonym)
                
                if synonyms:
                    new_token = random.choice(synonyms)
                    if token[0].isupper():
                        new_token = new_token.capitalize()
                    transformed_tokens.append(new_token)
                    continue
        
        if random.random() < 0.10 and len(token) > 2:
            token_lower = token.lower()
            edge_positions = [0, len(token) - 1]
            pos = random.choice(edge_positions)
            char = token_lower[pos]
            if char in qwerty_map and qwerty_map[char]:
                # Replace with a neighboring key
                replacement = random.choice(qwerty_map[char])
                # Preserve original case
                if token[pos].isupper():
                    replacement = replacement.upper()
                new_token = token[:pos] + replacement + token[pos+1:]
                transformed_tokens.append(new_token)
                continue
        
        # Randomly capitalize or lowercase words with 20% probability
        if random.random() < 0.20:
            if random.random() < 0.5:
                # Convert to lowercase
                new_token = token.lower()
            else:
                # Convert to uppercase
                new_token = token.upper()
            transformed_tokens.append(new_token)
            continue
        
        transformed_tokens.append(token)
    
    # Reconstruct the text
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
