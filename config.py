DATA_PATH = "../data"

MOVIE_PATH = f"{DATA_PATH}/Movie"
BOOKS_PATH = f"{DATA_PATH}/Books"
ELECTRONICS_PATH = f"{DATA_PATH}/Electronics"
SPORTS_PATH = f"{DATA_PATH}/Sports"

MOVIE_FINAL_DATA_PATH = f"{MOVIE_PATH}/final_data.jsonl"
MOVIE_CONVERSATION_PATH = f"{MOVIE_PATH}/Conversation.txt"
MOVIE_USER_MAP_PATH = f"{MOVIE_PATH}/user_ids.json"
MOVIE_ITEM_MAP_PATH = f"{MOVIE_PATH}/item_map.json"

BOOKS_FINAL_DATA_PATH = f"{BOOKS_PATH}/final_data.jsonl"
BOOKS_CONVERSATION_PATH = f"{BOOKS_PATH}/Conversation.txt"
BOOKS_USER_MAP_PATH = f"{BOOKS_PATH}/user_ids.json"
BOOKS_ITEM_MAP_PATH = f"{BOOKS_PATH}/item_map.json"

ELECTRONICS_FINAL_DATA_PATH = f"{ELECTRONICS_PATH}/final_data.jsonl"
ELECTRONICS_CONVERSATION_PATH = f"{ELECTRONICS_PATH}/Conversation.txt"
ELECTRONICS_USER_MAP_PATH = f"{ELECTRONICS_PATH}/user_ids.json"
ELECTRONICS_ITEM_MAP_PATH = f"{ELECTRONICS_PATH}/item_map.json"

SPORTS_FINAL_DATA_PATH = f"{SPORTS_PATH}/final_data.jsonl"
SPORTS_CONVERSATION_PATH = f"{SPORTS_PATH}/Conversation.txt"
SPORTS_USER_MAP_PATH = f"{SPORTS_PATH}/user_ids.json"
SPORTS_ITEM_MAP_PATH = f"{SPORTS_PATH}/item_map.json"

CATEGORIES = {
    "Movies": {
        "final_data": MOVIE_FINAL_DATA_PATH,
        "conversation": MOVIE_CONVERSATION_PATH,
        "user_map": MOVIE_USER_MAP_PATH,
        "item_map": MOVIE_ITEM_MAP_PATH,
    },
    "Books": {
        "final_data": BOOKS_FINAL_DATA_PATH,
        "conversation": BOOKS_CONVERSATION_PATH,
        "user_map": BOOKS_USER_MAP_PATH,
        "item_map": BOOKS_ITEM_MAP_PATH,
    },
    "Electronics": {
        "final_data": ELECTRONICS_FINAL_DATA_PATH,
        "conversation": ELECTRONICS_CONVERSATION_PATH,
        "user_map": ELECTRONICS_USER_MAP_PATH,
        "item_map": ELECTRONICS_ITEM_MAP_PATH,
    },
    "Sports": {
        "final_data": SPORTS_FINAL_DATA_PATH,
        "conversation": SPORTS_CONVERSATION_PATH,
        "user_map": SPORTS_USER_MAP_PATH,
        "item_map": SPORTS_ITEM_MAP_PATH,
    },
}