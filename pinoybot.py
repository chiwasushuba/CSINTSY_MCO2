
import os
import pickle
from typing import List, Dict, Any

# Define the model filename
MODEL_FILE = 'pinoybot_model.pkl'

def extract_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts features for each token in a list,
    now including contextual features.
    """
    feature_list = []
    
    # Pad tokens to safely check for previous/next words
    # _START_ and _END_ are special "padding" tokens
    padded_tokens = ['_START_'] + [t.lower() for t in tokens] + ['_END_']
    
    for i in range(1, len(padded_tokens) - 1):
        token = padded_tokens[i] # The current token
        original_token_case = tokens[i-1] # Get original case from non-padded list
        
        # Get context (previous and next words, in lowercase)
        prev_token = padded_tokens[i-1]
        next_token = padded_tokens[i+1]

        features = {
            # --- Word-level features ---
            'len': len(token),
            'is_upper': original_token_case.isupper(),
            'is_title': original_token_case.istitle(),
            'is_lower': original_token_case.islower(),
            'vowel_ratio': sum(1 for c in token if c in 'aeiou') / (len(token) + 0.01),
            'is_numeric': token.isdigit(),
            'is_alpha': token.isalpha(),

            # --- Affix features (add more!) ---
            'ends_in_ng': token.endswith('ng'),
            'starts_with_ma': token.startswith('ma'),
            'starts_with_nag': token.startswith('nag'),
            'ends_in_s': token.endswith('s'),
            'ends_in_ed': token.endswith('ed'),
            'ends_in_ing': token.endswith('ing'),

            # --- Character-level features ---
            'has_hyphen': '-' in token,
            'has_apostrophe': "'" in token,
            'has_double_e': 'ee' in token, # Good for 'meet'
            'has_double_o': 'oo' in token,
            'has_th': 'th' in token,
            
            # --- REAL Context features ---
            'prev_word': prev_token,
            'prev_word_is_start': prev_token == '_START_',
            'prev_word_is_i': prev_token == 'i',
            'prev_word_is_sa': prev_token == 'sa',
            
            'next_word': next_token,
            'next_word_is_end': next_token == '_END_'
        }
        feature_list.append(features)
    
    return feature_list

def load_model(model_path: str):
    """Loads the bundled model and vectorizer from disk."""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run 'train_model.py' first to create the model file.")
        return None, None
        
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    
    return bundle['model'], bundle['vectorizer']

model, vectorizer = load_model(MODEL_FILE)

def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    
    Args:
        tokens: List of word tokens (strings).
    
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    
    if model is None or vectorizer is None:
        return ['OTH'] * len(tokens)

    features_dict_list = extract_features(tokens)

    X_vectorized = vectorizer.transform(features_dict_list)

    predicted_tags = model.predict(X_vectorized)

    tags = list(predicted_tags)
    
    return tags

if __name__ == "__main__":
    
    if model is not None:
        example_tokens = ["Love", "kita", "."]
        print("Tokens:", example_tokens)
        tags = tag_language(example_tokens)
        print("Tags:", tags) # Expected: ['ENG', 'FIL', 'OTH']
        
        example_tokens_2 = ["nag", "lunch", "na", "sa", "DLSU"]
        print("\nTokens:", example_tokens_2)
        tags_2 = tag_language(example_tokens_2)
        print("Tags:", tags_2) # Expected: ['FIL', 'ENG', 'FIL', 'FIL', 'ENG'] (based on dummy data)

        example_tokens_3 = ["I", "will", "meet", "you", "sa", "park"]
        print("\nTokens:", example_tokens_3)
        tags_3 = tag_language(example_tokens_3)
        print("Tags:", tags_3) 
    
    else:
        print("\nCannot run example: Model not loaded.")