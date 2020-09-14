import os

import nltk
import sys
import string
from nltk.corpus import stopwords
import math
from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))
    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_content=dict()
    for filename in os.listdir(directory):
        content=""
        with open(os.path.join(directory, filename),encoding='UTF-8',newline='\n') as f:
            content+=f.read().lower()

        file_content.update({filename:content})
    return file_content
def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    content=nltk.word_tokenize(document.lower())
    stop_words = set(stopwords.words('english'))
    filtered_content=[w for w in content if not w in stop_words and not w in string.punctuation]
    return filtered_content


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs=dict()
    words=set()
    for filename in documents:
        words.update(documents[filename])
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = dict()
    top_file_list=list()
    for filename in files:
        weight_query=0
        for word in query:
            if word in files[filename]:
                tf = Counter(files[filename])
                weight_query+=(tf[word] * idfs[word])
        tfidfs[filename]=weight_query

    sorted_filenames = sorted(tfidfs.items(), key=lambda x: x[1], reverse=True)
    final_query_filenames=sorted_filenames[0: n]
    for file in final_query_filenames:
        top_file_list.append(file[0])
    return top_file_list


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf_score=dict()
    final_sentence=list()
    for sentence in sentences:
        words=set(sentences[sentence])
        idf_sum=0
        term_freq=0
        word_idf=list()
        for word in words:
            if word in query:
                idf_sum+=idfs[word]
                word_idf.append((word,idfs[word]))
                term_freq+=1
        term_density=term_freq/len(words)
        idf_score[sentence]={"idf":idf_sum,"density":term_density,"word_idf":word_idf}
    sorted_sentences=sorted(idf_score.items(), key=lambda x: (-x[1]["idf"], -x[1]["density"]))
    for sentence in sorted_sentences[0: n]:
        final_sentence.append(sentence[0][0].upper()+sentence[0][1:])

    return final_sentence


if __name__ == "__main__":
    main()
