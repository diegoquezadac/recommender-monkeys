import polars as pl
import json
from scripts.Tools import read_json, read_jsonl, read_dialogue, split_dialogues
import config
from typing import Dict, List, Any


class DataLoader:
    """
    A class to load data from various categories (Movies, Books, Electronics, Sports) 
    in the LLM-Redial dataset, convert them into Polars DataFrames, and store them.
    """

    def __init__(self) -> None:
        """
        Initializes the DataLoader with an empty dictionary
        to store the loaded data as Polars DataFrames.
        """
        self.dataframes: Dict[str, Dict[str, pl.DataFrame]] = {}

    def _parse_conversation(self, conversation_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Parses and formats the conversation data for each user.

        Args:
            conversation_data (List[Dict[str, Any]]): The list of conversation objects.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing formatted conversation data.
        """
        conversations = {}
        for j, conv in enumerate(conversation_data):
            conversation_key = f"conversation_{j + 1}"
            per_conversation_info = conv[conversation_key]

            conversation_id = per_conversation_info.get("conversation_id")
            user_likes = per_conversation_info.get("user_likes", [])
            user_dislikes = per_conversation_info.get("user_dislikes", [])
            rec_item = per_conversation_info.get("rec_item", [])

            conversations[conversation_key] = {
                "conversation_id": conversation_id,
                "user_likes": user_likes,
                "user_dislikes": user_dislikes,
                "rec_item": rec_item
            }

        return conversations

    def load_data(self) -> Dict[str, Dict[str, pl.DataFrame]]:
        """
        Loads the data for each category, converts it into Polars DataFrames, 
        and stores it in a dictionary.

        Returns:
            Dict[str, Dict[str, pl.DataFrame]]: A dictionary containing the dataframes 
            for each category and data type.
        """
        for category, paths in config.CATEGORIES.items():
            print(f"Loading data for {category}...")

            item_map = read_json(paths["item_map"])
            user_map = read_json(paths["user_map"])

            df_item_map = pl.DataFrame([{"item_id": k, "item_name": v} for k, v in item_map.items()])
            df_user_map = pl.DataFrame([{"user_id": k, "user_info": v} for k, v in user_map.items()])

            final_data = read_jsonl(paths["final_data"])

            parsed_data = []
            for line in final_data:
                data = json.loads(line)
                user_id, user_info = next(iter(data.items()))
                history_interaction = user_info.get("history_interaction", [])
                user_might_like = user_info.get("user_might_like", [])
                conversations = self._parse_conversation(user_info.get("Conversation", []))

                parsed_data.append({
                    "user_id": user_id,
                    "history_interaction": history_interaction,
                    "user_might_like": user_might_like,
                    "conversations": conversations
                })

            df_final_data = pl.DataFrame(parsed_data)

            dialogue_content = read_dialogue(paths["conversation"])

            dialogues = split_dialogues(dialogue_content)

            df_dialogues = pl.DataFrame([{"conversation_id": int(conv_id), "dialogue": text} for conv_id, text in dialogues])

            self.dataframes[category] = {
                "item_map": df_item_map,
                "user_map": df_user_map,
                "final_data": df_final_data,
                "dialogues": df_dialogues
            }

            print(f"Data for {category} loaded successfully.\n")

        return self.dataframes
