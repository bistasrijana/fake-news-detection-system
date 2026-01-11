import numpy as np

def soft_voting(scores_dict):
    """
    scores_dict = {
      "LinearSVC": 0.82,
      "Logistic Regression": 0.91,
      "HAN": 0.88
    }
    """

    available_scores = [v for v in scores_dict.values() if v is not None]

    if len(available_scores) == 0:
        return None, "No model produced a score."

    
    combined_score = float(np.mean(available_scores))

    
    label = "REAL" if combined_score >= 0.5 else "FAKE"

    return combined_score, label
