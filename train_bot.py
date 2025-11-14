import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from typing import List, Dict, Any

def extract_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts features for each token in a list.
    This is a basic example. You MUST expand this based on your
    feature engineering.
    """
    feature_list = []
    for i, token in enumerate(tokens):
        features = {
            # --- Word-level features ---
            'len': len(token),
            'is_upper': token.isupper(),
            'is_title': token.istitle(),
            'is_lower': token.islower(),
            'vowel_ratio': sum(1 for c in token.lower() if c in 'aeiou') / (len(token) + 0.01),
            
            # --- Affix features (basic) ---
            'ends_in_ng': token.endswith('ng'),
            'starts_with_ma': token.startswith('ma'),
            'starts_with_nag': token.startswith('nag'),

            # --- Character-level features [cite: 70] ---
            'has_hyphen': '-' in token,
            'has_apostrophe': "'" in token,
            
            # --- Context features (properties of surrounding words) [cite: 71] ---
            'prev_word_is_fil': False, # You would set this in a more complex loop
            'next_word_is_eng': False, # You would set this in a more complex loop
        }
        feature_list.append(features)
        
    return feature_list

def map_to_final_tags(annotated_tag: str) -> str:
    """
    Maps the detailed annotation tags to the three final classes:
    ENG, FIL, or OTH, based on Section 2.3 specifications.
    """
    if annotated_tag in ['FIL', 'CS']:
        return 'FIL'
    
    if annotated_tag == 'ENG':
        return 'ENG'
    
    if 'ENG' in annotated_tag:
        return 'ENG'
    if 'FIL' in annotated_tag:
        return 'FIL'

    return 'OTH'

print("Loading data from Excel file...")
df = pd.read_excel('Dataset/MCO2 Dataset (full).xlsx')

df['final_label'] = df['corrected_label'].fillna(df['label'])

# Remove rows with missing word or label
df = df.dropna(subset=['word', 'final_label'])

# Convert to string to handle any non-string values
df['word'] = df['word'].astype(str)
df['final_label'] = df['final_label'].astype(str)

# Group by sentence_id to reconstruct sentences
raw_data = []
for sentence_id, group in df.groupby('sentence_id'):
    tokens = group['word'].tolist()
    tags = group['final_label'].tolist()
    raw_data.append((tokens, tags))

print(f"Loaded {len(raw_data)} sentences from dataset")

# Flatten data into single lists of tokens and mapped labels
all_tokens = []
all_labels = []

for tokens, tags in raw_data:
    all_tokens.extend(tokens)
    mapped_tags = [map_to_final_tags(tag) for tag in tags]
    all_labels.extend(mapped_tags)

print(f"Total tokens processed: {len(all_tokens)}")

from sklearn.feature_extraction import DictVectorizer

print("Extracting features...")
# We must use DictVectorizer because scikit-learn models need numerical input
vectorizer = DictVectorizer(sparse=False)
X_features = extract_features(all_tokens)
X_vectorized = vectorizer.fit_transform(X_features)

y = all_labels

print("Splitting data...")
# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

print("Training model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("\n--- Model Evaluation (Test Set) ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("-----------------------------------\n")


model_bundle = {
    'model': model,
    'vectorizer': vectorizer
}

MODEL_FILE = 'pinoybot_model.pkl'
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model_bundle, f)

print(f"Model and vectorizer saved to '{MODEL_FILE}'")