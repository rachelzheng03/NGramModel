import sys
import json
import pandas as pd
import time
from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

def print_progress(percent):
    width = 50  # Width of the progress bar
    filled = int(width * percent / 100)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {percent}%")
    sys.stdout.flush()

def load_model_data(filename:str):
    with open(filename, "r") as f:
        loaded_data = json.load(f)
    return loaded_data

def tokenize_methods_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Tokenize the methods in method_column

    Args:
        df (pd.DataFrame): DataFrame containing the methods.
        method_column (str): Column name containing the Java methods.
        language (str): Programming language for the lexer (e.g., 'java').

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Tokenized Method'.
    """
    train_size = len(df[method_column])
    count=0
    def tokenize_method(code):
        lexer = get_lexer_by_name(language)
        tokens = [t[1] for t in lexer.get_tokens(code) if t[0] not in Token.Text]
        nonlocal count
        count += 1

        if count%200 == 0:
            print_progress(int((count/train_size)*100))
            time.sleep(0.05)

        return tokens

    # Apply the function to the specified column and add a new column with the results
    df["Tokenized Method"] = df[method_column].apply(tokenize_method)
    print_progress(100)
    print()
    return df