import numpy as np

def recall_at_k(relevant_items, recommended_items, k):

    if not relevant_items:
        return 0.0  # Avoid division by zero when there are no relevant items

    recommended_at_k = set(recommended_items[:k])
    hits = recommended_at_k.intersection(relevant_items)
    
    return len(hits) / len(relevant_items)

def ndcg_at_k(relevant_items, recommended_items, k):

    def dcg(scores):
        return np.sum([
            score / np.log2(idx + 2)  # idx + 2 because indices are 0-based
            for idx, score in enumerate(scores)
        ])
    
    # Generate relevance scores for recommended items
    relevance = [1 if item in relevant_items else 0 for item in recommended_items[:k]]
    ideal_relevance = sorted(relevance, reverse=True)
    
    dcg_score = dcg(relevance)
    idcg_score = dcg(ideal_relevance)
    
    return dcg_score / idcg_score if idcg_score > 0 else 0.0

def novelty_at_k(recommended_items, item_interactions, k):
    # To avoid division by zero, we can add a small constant (epsilon) to the interaction frequencies
    epsilon = 1e-5
    
    novelty_score = 0
    for item in recommended_items[:k]:
        # Get the interaction frequency for the item, use epsilon if the item is not in item_interactions
        freq = item_interactions.get(item, epsilon)
        
        # Add the inverse of the frequency to the novelty score (higher frequency, lower novelty)
        novelty_score += 1 / freq
    
    # Return the average novelty score over the top k recommended items
    return novelty_score / k

def precision_at_k(relevant_items, recommended_items, k):
    
    if k == 0:
        return 0.0  # Avoid division by zero
    
    recommended_at_k = recommended_items[:k]
    hits = set(recommended_at_k).intersection(relevant_items)
    
    return len(hits) / k
