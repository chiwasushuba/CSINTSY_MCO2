import pickle
import os

# Global model variable
model_pipeline = None

def load_model():
    global model_pipeline
    model_path = 'pinoybot_model.pkl'

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_pipeline = pickle.load(f)
        print("Model loaded successfully.")
    else:
        print(f"Error: {model_path} not found. Please run train_bot.py first.")
        model_pipeline = None

def strong_english_indicators(word):
    """Strong rule-based English indicators"""
    word_lower = word.lower()
    
    strong_indicators = [
        word_lower.endswith('ing'), word_lower.endswith('ed'), word_lower.endswith('tion'),
        word_lower.endswith('sion'), word_lower.endswith('ment'), word_lower.endswith('ness'),
        word_lower.endswith('ity'), word_lower.endswith('ive'), word_lower.endswith('ous'),
        word_lower.endswith('able'), word_lower.endswith('ible'), word_lower.endswith('ly'),
        word_lower.endswith('er'), word_lower.endswith('est'), word_lower.endswith('ful'),
        word_lower.startswith('un'), word_lower.startswith('re'), word_lower.startswith('pre'),
        word_lower.startswith('dis'), word_lower.startswith('mis'), word_lower.startswith('non'),
        word_lower.endswith('ove'), word_lower.endswith('upt'), word_lower.endswith('ine'),
        word_lower.endswith('all'), word_lower == 'the', word_lower == 'go', 
        word_lower == 'about', "let's" in word_lower,
        'th' in word_lower, 'sh' in word_lower, 'ch' in word_lower, 'wh' in word_lower,
        'ph' in word_lower, 'gh' in word_lower, 'ck' in word_lower,
    ]
    
    return sum(strong_indicators)

def strong_filipino_indicators(word):
    """Strong rule-based Filipino indicators"""
    word_lower = word.lower()
    
    strong_indicators = [
        word_lower.startswith('mag'), word_lower.startswith('nag'), word_lower.startswith('pag'),
        word_lower.startswith('pang'), word_lower.startswith('mang'), word_lower.startswith('nang'),
        word_lower.startswith('ka'), word_lower.startswith('pa'), word_lower.startswith('ma'),
        word_lower.startswith('na'), word_lower.startswith('um'),
        word_lower.endswith('ng'), word_lower.endswith('an'), word_lower.endswith('in'),
        word_lower.endswith('han'), word_lower.endswith('hin'), word_lower.endswith('ito'),
        word_lower in ['ang', 'ng', 'sa', 'mga', 'na', 'ay', 'kami', 'namin', 'kita', 'ako'],
        len(word_lower) > 4 and word_lower[:2] == word_lower[2:4],
    ]
    
    return sum(strong_indicators)

def extract_features_smart(tokens):
    """Smart feature extraction matching train_bot.py"""
    feature_list = []
    padded = ['_START_'] + tokens + ['_END_']

    for i in range(1, len(padded) - 1):
        token = padded[i]
        prev_word = padded[i-1].lower()
        next_word = padded[i+1].lower()
        word_lower = token.lower()

        if not token.isalpha():
            features = {
                'is_symbol': not token.isalnum(),
                'is_number': token.isdigit(),
                'len': len(token),
            }
            feature_list.append(features)
            continue

        eng_strength = strong_english_indicators(token)
        fil_strength = strong_filipino_indicators(token)
        
        vowels = sum(1 for c in word_lower if c in 'aeiou')
        vowel_ratio = vowels / len(word_lower) if len(word_lower) > 0 else 0
        
        char_trigrams = [word_lower[i:i+3] for i in range(len(word_lower)-2)] if len(word_lower) >= 3 else []
        
        features = {
            'english_strength': eng_strength,
            'filipino_strength': fil_strength,
            'strength_difference': eng_strength - fil_strength,
            'len': len(token),
            'vowel_ratio': vowel_ratio,
            'starts_with_vowel': word_lower[0] in 'aeiou' if word_lower else False,
            'has_english_clusters': any(cluster in word_lower for cluster in ['th', 'sh', 'ch', 'wh', 'ph', 'gh']),
            'has_rare_letters': any(letter in word_lower for letter in 'fvxz'),
            'english_trigram_count': sum(1 for tg in char_trigrams if tg in ['ing', 'ion', 'ent', 'ess', 'ble', 'ive', 'ous']),
            'filipino_trigram_count': sum(1 for tg in char_trigrams if tg in ['ang', 'ng', 'mag', 'nag', 'pag', 'han', 'hin', 'ito']),
            'has_double_vowel': any(word_lower[i] == word_lower[i+1] and word_lower[i] in 'aeiou' for i in range(len(word_lower)-1)),
            'prev_is_filipino': prev_word in ['ang', 'ng', 'sa', 'mga', 'na', 'ay'],
            'next_is_filipino': next_word in ['ang', 'ng', 'sa', 'mga', 'na', 'ay'],
            'prev_is_english': prev_word in ['the', 'to', 'of', 'and', 'is', 'in', 'on', 'at'],
            'next_is_english': next_word in ['the', 'to', 'of', 'and', 'is', 'in', 'on', 'at'],
            'has_hyphen': '-' in token,
            'has_apostrophe': "'" in token,
        }
        
        feature_list.append(features)
    
    return feature_list

def tag_language(tokens):
    global model_pipeline

    if model_pipeline is None:
        load_model()

    if model_pipeline is None:
        return ['OTH'] * len(tokens)

    try:
        if not tokens or len(tokens) == 0:
            return []
            
        features = extract_features_smart(tokens)
        predicted_tags = model_pipeline.predict(features)
        predicted_tags = [str(tag) for tag in predicted_tags]
        
        if len(predicted_tags) != len(tokens):
            return ['OTH'] * len(tokens)
            
        return predicted_tags
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ['OTH'] * len(tokens)

def test_implementation():
    test_cases = [
        ["Love", "kita", "."],
        ["Bawal", "ang", "corrupt", "dito", "."],
        ["Nag-lunch", "kami", "sa", "Jollibee", "kahapon", "."],
        ["Ang", "thesis", "namin", "ay", "about", "Machine", "Learning", "."],
        ["Let's", "go", "to", "the", "mall", "ng", "bukas", "!"],
        ["123", "abc", "!", "hello"]
    ]

    print("Testing PinoyBot Implementation...")
    print("=" * 50)
    
    for i, tokens in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {tokens}")
        tags = tag_language(tokens)
        print(f"Output: {tags}")

if __name__ == "__main__":
    test_implementation()
