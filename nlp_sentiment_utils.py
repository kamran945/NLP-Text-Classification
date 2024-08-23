from collections import Counter
import pandas as pd


def get_vocabulary(df: pd.DataFrame, column: str = "text") -> Counter:
    """
    This function calculates the vocabulary size of the text data in a pandas DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        column (str, optional): The name of the column containing the text data. Defaults to "text".
    Returns:
        Counter: A Counter object containing the vocabulary size of each word.
    """

    # Combine text from multiple columns into one
    vocabulary = Counter(" ".join(df[column]).split())

    print(f"Total words in vocabulary: {len(vocabulary)}\n")
    print(f"Most common 50 words in vocabulary:\n {vocabulary.most_common(50)}")

    return vocabulary


import numpy as np


def reduce_vocabulary(vocabulary: Counter, quantile: float = 0.95) -> Counter:
    """
    This function reduces the vocabulary size of the text data in a pandas DataFrame.
    Args:
        vocabulary (Counter): A Counter object containing the vocabulary and count values of words.
        quantile (float, optional): The quantile threshold for reducing the vocabulary size. Defaults to 0.95.
    Returns:
        Counter: A Counter object containing the reduced vocabulary.
    """

    frequencies = np.array(list(vocabulary.values()))
    quantile_vlaue = np.quantile(frequencies, quantile)
    reduced_vocab = Counter(
        {word: count for word, count in vocabulary.items() if count >= quantile_vlaue}
    )

    print(f"Total words in previous vocabulary: {len(vocabulary)}")
    print(f"Total words in reduced vocabulary: {len(reduced_vocab)}\n")
    print(
        f"Most common 50 words in reduced vocabulary:\n {reduced_vocab.most_common(50)}"
    )

    return reduced_vocab


import pandas as pd
from nltk.corpus import stopwords
import contractions
from bs4 import BeautifulSoup
from textblob import TextBlob


def clean_text(df: pd.DataFrame, column: str = "text") -> pd.DataFrame:
    """
    This function cleans the text data in a pandas DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        column (str, optional): The name of the column containing the text data. Defaults to "text".
    Returns:
        pd.DataFrame: The cleaned DataFrame with cleaned text data.
    """
    # Lowercasing
    #     df[column] = df[column].str.lower()

    # Remove Punctuation
    df[column] = df[column].str.replace("[^\w\s]", "", regex=True)

    # Remove Numbers
    df[column] = df[column].str.replace("\d+", "", regex=True)

    # Remove Special Characters
    df[column] = df[column].str.replace("[^A-Za-z0-9 ]+", "", regex=True)

    # Remove Extra Whitespace
    df[column] = df[column].str.strip()
    df[column] = df[column].str.replace("\s+", " ", regex=True)

    # Remove Stop Words
    stop = stopwords.words("english")
    df[column] = df[column].apply(
        lambda x: " ".join([word for word in x.split() if word not in stop])
    )

    # Handle Contractions
    df[column] = df[column].apply(lambda x: contractions.fix(x))

    # Remove URLs
    df[column] = df[column].str.replace("http\S+|www.\S+", "", regex=True)

    # Remove HTML Tags
    df[column] = df[column].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())

    #     # Correct Spelling (Optional)
    #     df[column] = df[column].apply(lambda x: str(TextBlob(x).correct()))

    return df


from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


def plot_wordcloud(vocabulary: Counter) -> None:
    """
    This function plots a word cloud based on the vocabulary.
    Args:
        vocabulary (Counter): A Counter object containing the vocabulary and count values of words.
    """
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(vocabulary)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np


def get_eval_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """
    This function calculates evaluation metrics for a binary classification task.
    Args:
        y_true (np.array): The true labels.
        y_pred (np.array): The predicted labels.
    Returns:
        dict: A dictionary containing the accuracy, precision, recall, and F1 score.
    """
    # Calculate precision, recall, and F1 score
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    eval_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

    return eval_metrics
