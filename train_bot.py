import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import re

# --- Configuration ---
DATASET_FILE = 'TaggedDataset.xlsx'
OTHER_DATASET_FILE = 'OtherDataset.xlsx' 
MODEL_FILE = 'pinoybot_model.pkl'

def get_final_label(row):
    is_correct = row.get('is_correct')
    corrected = row.get('corrected_label')
    s_correct = str(is_correct).strip().lower()
    
    if s_correct in ['no', 'false', '0', 'f']:
        if pd.notna(corrected) and str(corrected).strip() != '':
            return str(corrected).strip()
    return str(row['label']).strip()

def map_to_final_tags(tag):
    tag = str(tag).upper().strip()
    if tag in ['FIL', 'TAG', 'TL']: return 'FIL'
    if tag in ['ENG', 'EN', 'ENGLISH']: return 'ENG'
    if tag == 'CS': return 'FIL'
    if tag in ['NE', 'NUM', 'SYM', 'ABB', 'EXPR', 'UNK']: return 'OTH'
    return 'OTH'

def strong_english_indicators(word):
    """Strong rule-based English indicators"""
    word_lower = word.lower()
    
    # English patterns
    strong_indicators = [
        # English suffixes
        word_lower.endswith('ing'), word_lower.endswith('ed'), word_lower.endswith('tion'),
        word_lower.endswith('sion'), word_lower.endswith('ment'), word_lower.endswith('ness'),
        word_lower.endswith('ity'), word_lower.endswith('ive'), word_lower.endswith('ous'),
        word_lower.endswith('able'), word_lower.endswith('ible'), word_lower.endswith('ly'),
        word_lower.endswith('er'), word_lower.endswith('est'), word_lower.endswith('ful'),

        word_lower.endswith('ove'), 
        word_lower.endswith('upt'),
        word_lower.endswith('ine'),
        word_lower.endswith('all'),
        word_lower == 'the', word_lower == 'go', word_lower == 'about',
        "let's" in word_lower,
        
        # English prefixes
        word_lower.startswith('un'), word_lower.startswith('re'), word_lower.startswith('pre'),
        word_lower.startswith('dis'), word_lower.startswith('mis'), word_lower.startswith('non'),
        word_lower.startswith('over'), word_lower.startswith('under'), word_lower.startswith('inter'),
        
        # English character patterns
        'th' in word_lower, 'sh' in word_lower, 'ch' in word_lower, 
        'wh' in word_lower, 'ph' in word_lower, 'gh' in word_lower,
        'ck' in word_lower, 'ough' in word_lower, 'eigh' in word_lower,
    ]
    
    return sum(strong_indicators)

def strong_filipino_indicators(word):
    """Strong rule-based Filipino indicators"""
    word_lower = word.lower()
    
    # Filipino patterns
    strong_indicators = [
        # Filipino affixes
        word_lower.startswith('mag'), word_lower.startswith('nag'), word_lower.startswith('pag'),
        word_lower.startswith('pang'), word_lower.startswith('mang'), word_lower.startswith('nang'),
        word_lower.startswith('ka'), word_lower.startswith('pa'), word_lower.startswith('ma'),
        word_lower.startswith('na'), word_lower.startswith('um'),
        
        # Filipino suffixes
        word_lower.endswith('ng'), word_lower.endswith('an'), word_lower.endswith('in'),
        word_lower.endswith('han'), word_lower.endswith('hin'), word_lower.endswith('ito'),
        word_lower.endswith('ate'), word_lower.endswith('ada'), word_lower.endswith('ido'),
        
        # Common Filipino words
        word_lower in ['ang', 'ng', 'sa', 'mga', 'na', 'ay', 'kami', 'namin', 'kita',
                      'ako', 'ikaw', 'siya', 'tayo', 'kayo', 'sila', 'ito', 'iyan',
                      'dito', 'doon', 'kung', 'kapag', 'pero', 'at', 'o', 'ni', 'kay'],
        
        # Reduplication
        len(word_lower) > 4 and word_lower[:2] == word_lower[2:4],
    ]
    
    return sum(strong_indicators)

def extract_features_smart(tokens):
    """Smart feature extraction that uses rule-based pre-filtering"""
    feature_list = []
    padded = ['_START_'] + tokens + ['_END_']

    for i in range(1, len(padded) - 1):
        token = padded[i]
        prev_word = padded[i-1].lower()
        next_word = padded[i+1].lower()
        word_lower = token.lower()

        # For non-alphabetic tokens
        if not token.isalpha():
            features = {
                'is_symbol': not token.isalnum(),
                'is_number': token.isdigit(),
                'len': len(token),
            }
            feature_list.append(features)
            continue

        # Calculate strong indicators
        eng_strength = strong_english_indicators(token)
        fil_strength = strong_filipino_indicators(token)
        
        # Basic character features
        vowels = sum(1 for c in word_lower if c in 'aeiou')
        vowel_ratio = vowels / len(word_lower) if len(word_lower) > 0 else 0
        
        # Character n-grams
        char_trigrams = [word_lower[i:i+3] for i in range(len(word_lower)-2)] if len(word_lower) >= 3 else []
        
        features = {
            # Rule based strengths
            'english_strength': eng_strength,
            'filipino_strength': fil_strength,
            'strength_difference': eng_strength - fil_strength,
            
            # Basic characteristics
            'len': len(token),
            'vowel_ratio': vowel_ratio,
            'starts_with_vowel': word_lower[0] in 'aeiou' if word_lower else False,
            
            # English character patterns
            'has_english_clusters': any(cluster in word_lower for cluster in ['th', 'sh', 'ch', 'wh', 'ph', 'gh']),
            'has_rare_letters': any(letter in word_lower for letter in 'fvxz'),
            'english_trigram_count': sum(1 for tg in char_trigrams if tg in ['ing', 'ion', 'ent', 'ess', 'ble', 'ive', 'ous']),
            
            # Filipino character patterns
            'filipino_trigram_count': sum(1 for tg in char_trigrams if tg in ['ang', 'ng', 'mag', 'nag', 'pag', 'han', 'hin', 'ito']),
            'has_double_vowel': any(word_lower[i] == word_lower[i+1] and word_lower[i] in 'aeiou' for i in range(len(word_lower)-1)),
            
            # Context features
            'prev_is_filipino': prev_word in ['ang', 'ng', 'sa', 'mga', 'na', 'ay'],
            'next_is_filipino': next_word in ['ang', 'ng', 'sa', 'mga', 'na', 'ay'],
            'prev_is_english': prev_word in ['the', 'to', 'of', 'and', 'is', 'in', 'on', 'at'],
            'next_is_english': next_word in ['the', 'to', 'of', 'and', 'is', 'in', 'on', 'at'],
            
            # Structural features
            'has_hyphen': '-' in token,
            'has_apostrophe': "'" in token,
        }
        
        feature_list.append(features)
    
    return feature_list

def create_balanced_dataset(df):
    """Create a more balanced dataset by oversampling English examples"""
    # Separate classes
    eng_df = df[df['target'] == 'ENG']
    fil_df = df[df['target'] == 'FIL']
    oth_df = df[df['target'] == 'OTH']
    
    print(f"Original: ENG={len(eng_df)}, FIL={len(fil_df)}, OTH={len(oth_df)}")
    
    # Oversample English to match at least 30% of Filipino
    target_eng_size = max(len(eng_df), int(0.3 * len(fil_df)))
    
    if len(eng_df) < target_eng_size:
        # Repeat English samples
        repeat_factor = target_eng_size // len(eng_df) + 1
        eng_df_oversampled = pd.concat([eng_df] * repeat_factor, ignore_index=True)
        eng_df_oversampled = eng_df_oversampled.head(target_eng_size)
        print(f"Oversampled ENG from {len(eng_df)} to {len(eng_df_oversampled)}")
    else:
        eng_df_oversampled = eng_df
    
    # Combine back
    balanced_df = pd.concat([eng_df_oversampled, fil_df, oth_df], ignore_index=True)
    print(f"Balanced: ENG={len(eng_df_oversampled)}, FIL={len(fil_df)}, OTH={len(oth_df)}")
    
    return balanced_df

def main():
    print("Loading datasets with SMART approach...")

    # Load datasets
    df1 = pd.read_excel(DATASET_FILE)
    df2 = pd.read_excel(OTHER_DATASET_FILE)
    df = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Combined dataset: {len(df)} total rows")

    # Clean data
    if 'is_dirty' in df.columns:
        df = df[df['is_dirty'] != True]

    # Process labels
    print("Processing labels...")
    df['resolved_label'] = df.apply(get_final_label, axis=1)
    df['target'] = df['resolved_label'].apply(map_to_final_tags)
    
    # Create balanced dataset
    df_balanced = create_balanced_dataset(df)
    
    # Split by sentences
    sentence_ids = df_balanced['sentence_id'].unique()
    
    train_val_sentences, test_sentences = train_test_split(
        sentence_ids, test_size=0.15, random_state=42
    )
    train_sentences, val_sentences = train_test_split(
        train_val_sentences, test_size=0.15/0.85, random_state=42
    )
    
    # Create splits
    train_df = df_balanced[df_balanced['sentence_id'].isin(train_sentences)]
    val_df = df_balanced[df_balanced['sentence_id'].isin(val_sentences)]
    test_df = df_balanced[df_balanced['sentence_id'].isin(test_sentences)]
    
    # Extract features
    def extract_from_df(dataframe):
        features, labels = [], []
        for _, group in dataframe.groupby('sentence_id'):
            words = group['word'].astype(str).tolist()
            tags = group['target'].tolist()
            sentence_features = extract_features_smart(words)
            features.extend(sentence_features)
            labels.extend(tags)
        return features, labels
    
    X_train, y_train = extract_from_df(train_df)
    X_val, y_val = extract_from_df(val_df)
    X_test, y_test = extract_from_df(test_df)
    
    print(f"Data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # Use aggressive class weighting
    class_counts = pd.Series(y_train).value_counts()
    class_weight_dict = {
        'ENG': class_counts.max() / class_counts.get('ENG', 1) * 2,  # Double weight for English
        'FIL': 1.0,
        'OTH': class_counts.max() / class_counts.get('OTH', 1)
    }
    print(f"Aggressive class weights: {class_weight_dict}")

    # Use simpler but more effective model
    pipeline = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\nTraining model with SMART features...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\nValidation Results:")
    y_pred_val = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred_val))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")

    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
