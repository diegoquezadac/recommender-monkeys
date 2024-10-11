import polars as pl
from nltk import word_tokenize, ngrams
from typing import Dict, Any


def compute_category_stats(category: str, dfs: Dict[str, pl.DataFrame]) -> Dict[str, Any]:
    """
    Computes statistics for a given category in the LLM-Redial dataset.

    Args:
        category (str): The name of the category (e.g., "Books", "Movies").
        dfs (Dict[str, pl.DataFrame]): A dictionary of Polars DataFrames for the category,
                                       containing 'final_data', 'dialogues', and 'item_map' DataFrames.

    Returns:
        Dict[str, Any]: A dictionary containing various statistics for the category including
                        number of dialogues, utterances, tokens, 4-grams, users, items, and averages.
    """
    df_dialogues = dfs["dialogues"]
    df_item_map = dfs["item_map"]
    df_user_map = dfs["user_map"]

    num_dialogues = df_dialogues.n_unique(subset="conversation_id")
    num_users = df_user_map.n_unique(subset="user_info")
    num_items = df_item_map.n_unique(subset="item_name")

    num_utterances = df_dialogues.select(pl.col("dialogue").count())[0, 0]

    total_tokens = sum(len(word_tokenize(dialogue)) for dialogue in df_dialogues["dialogue"])
    total_4grams = sum(len(list(ngrams(word_tokenize(dialogue), 4))) for dialogue in df_dialogues["dialogue"])

    avg_dialogues_per_user = num_dialogues / num_users if num_users > 0 else 0
    avg_utterances_per_dialogue = num_utterances / num_dialogues if num_dialogues > 0 else 0

    return {
        "Category": category,
        "#Dialogues": num_dialogues,
        "#Utterances": num_utterances,
        "#Tokens": total_tokens,
        "#4-Grams": total_4grams,
        "#Users": num_users,
        "#Items": num_items,
        "Avg. #Dialogues per User": round(avg_dialogues_per_user, 2),
        "Avg. #Utterances per Dialogue": round(avg_utterances_per_dialogue, 2)
    }


def split_dialogue_by_newline(df: pl.DataFrame) -> pl.DataFrame:
    """
    Splits the 'dialogue' column in the DataFrame by double newlines (\n\n)
    and removes any extra whitespace.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing a 'dialogue' column.

    Returns:
        pl.DataFrame: A modified DataFrame with the 'dialogue' column processed.
    """
    return df.with_columns(
        pl.col("dialogue")
          .map_elements(lambda dialogue: dialogue.strip().split("\n\n"), return_dtype=pl.List(pl.Utf8))
          .alias("dialogue")
    )
