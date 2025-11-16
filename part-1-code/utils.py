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

    # Common words to skip for synonym replacement
    common_words = {'the', 'a', 'an', 'he', 'she', 'i', 'you', 'it', 'we', 'they', 'him', 'her', 'his', 'hers', 'this'}

    tokens = word_tokenize(text)

    transformed_tkns = []

    for i, token in enumerate(tokens):
        if not token.isalpha():
            transformed_tkns.append(token)
            continue

        token_lw = token.lower()
        transformed_flg = False

        # 1. Synonym replacement (40% probability)
        if not transformed_flg and random.random() < 0.40 and token_lw not in common_words:
            synsets = wordnet.synsets(token_lw)
            synonyms = []

            if synsets:
                for syn in synsets:
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if (' ' not in synonym and synonym.isalpha() and 
                            synonym.lower() != token_lw and synonym.lower() != token.lower()):
                            synonyms.append(synonym)

            if synonyms:
                new_token = random.choice(synonyms)
                if token[0].isupper():
                    new_token = new_token.capitalize()
                transformed_tkns.append(new_token)
                transformed_flg = True

        if not transformed_flg:
            transformed_tkns.append(token)
    
    # Reconstruct the text with proper spacing
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_tkns)

    ##### YOUR CODE ENDS HERE ######

    return example