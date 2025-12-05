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

# --- 1. Configuration ---
DATASET_FILE = 'Dataset/MCO2 Dataset (full).xlsx' # Path to your dataset file
MODEL_FILE = 'pinoybot_model.pkl'

# --- 2. Helper Functions ---

# Common English words that should always be tagged as English
COMMON_ENGLISH_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'could', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for', 'on',
    'at', 'by', 'with', 'from', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'but', 'go', 'going', 'went', 'gone',
    'make', 'made', 'making', 'get', 'getting', 'got', 'take', 'taking',
    'took', 'taken', 'come', 'coming', 'came', 'know', 'knowing', 'knew',
    'think', 'thinking', 'thought', 'see', 'seeing', 'saw', 'seen',
    'want', 'wanted', 'wanting', 'use', 'using', 'used', 'find', 'finding',
    'found', 'give', 'giving', 'gave', 'given', 'tell', 'telling', 'told',
    'work', 'working', 'worked', 'call', 'calling', 'called', 'try',
    'trying', 'tried', 'ask', 'asking', 'asked', 'need', 'needing',
    'needed', 'feel', 'feeling', 'felt', 'become', 'becoming', 'became',
    'leave', 'leaving', 'left', 'put', 'putting', 'let', 'letting',
    'mean', 'meaning', 'meant', 'keep', 'keeping', 'kept', 'begin',
    'beginning', 'began', 'begun', 'seem', 'seeming', 'seemed', 'help',
    'helping', 'helped', 'show', 'showing', 'showed', 'shown', 'hear',
    'hearing', 'heard', 'play', 'playing', 'played', 'run', 'running',
    'ran', 'move', 'moving', 'moved', 'live', 'living', 'lived',
    'believe', 'believing', 'believed', 'bring', 'bringing', 'brought',
    'write', 'writing', 'wrote', 'written', 'sit', 'sitting', 'sat',
    'stand', 'standing', 'stood', 'lose', 'losing', 'lost', 'pay',
    'paying', 'paid', 'meet', 'meeting', 'met', 'include', 'including',
    'included', 'continue', 'continuing', 'continued', 'set', 'setting',
    'learn', 'learning', 'learned', 'learnt', 'change', 'changing',
    'changed', 'lead', 'leading', 'led', 'understand', 'understanding',
    'understood', 'watch', 'watching', 'watched', 'follow', 'following',
    'followed', 'stop', 'stopping', 'stopped', 'create', 'creating',
    'created', 'speak', 'speaking', 'spoke', 'spoken', 'read', 'reading',
    'allow', 'allowing', 'allowed', 'add', 'adding', 'added', 'spend',
    'spending', 'spent', 'grow', 'growing', 'grew', 'grown', 'open',
    'opening', 'opened', 'walk', 'walking', 'walked', 'win', 'winning',
    'won', 'offer', 'offering', 'offered', 'remember', 'remembering',
    'remembered', 'love', 'loving', 'loved', 'consider', 'considering',
    'considered', 'appear', 'appearing', 'appeared', 'buy', 'buying',
    'bought', 'serve', 'serving', 'served', 'die', 'dying', 'died',
    'send', 'sending', 'sent', 'build', 'building', 'built', 'stay',
    'staying', 'stayed', 'fall', 'falling', 'fell', 'fallen', 'cut',
    'cutting', 'reach', 'reaching', 'reached', 'kill', 'killing',
    'killed', 'raise', 'raising', 'raised', 'pass', 'passing', 'passed',
    'sell', 'selling', 'sold', 'decide', 'deciding', 'decided', 'return',
    'returning', 'returned', 'explain', 'explaining', 'explained',
    'hope', 'hoping', 'hoped', 'develop', 'developing', 'developed',
    'carry', 'carrying', 'carried', 'break', 'breaking', 'broke',
    'broken', 'receive', 'receiving', 'received', 'agree', 'agreeing',
    'agreed', 'support', 'supporting', 'supported', 'hit', 'hitting',
    'produce', 'producing', 'produced', 'eat', 'eating', 'ate', 'eaten',
    'cover', 'covering', 'covered', 'catch', 'catching', 'caught',
    'draw', 'drawing', 'drew', 'drawn', 'choose', 'choosing', 'chose',
    'chosen', 'mall', 'machine', 'computer', 'phone', 'internet',
    'facebook', 'google', 'youtube', 'email', 'website', 'online',
    'download', 'upload', 'click', 'search', 'post', 'share', 'like',
    'comment', 'message', 'chat', 'call', 'text', 'video', 'photo',
    'picture', 'music', 'game', 'app', 'software', 'hardware', 'data',
    'file', 'folder', 'document', 'report', 'project', 'assignment',
    'homework', 'test', 'exam', 'quiz', 'grade', 'score', 'class',
    'student', 'teacher', 'professor', 'school', 'college', 'university',
}

# Common Filipino words that should always be tagged as Filipino
COMMON_FILIPINO_WORDS = {
    'ang', 'ng', 'sa', 'mga', 'na', 'ay', 'at', 'ko', 'mo', 'niya',
    'natin', 'namin', 'ninyo', 'nila', 'ako', 'ikaw', 'siya', 'tayo',
    'kami', 'kayo', 'sila', 'ito', 'iyan', 'iyon', 'dito', 'diyan',
    'doon', 'nito', 'niyan', 'niyon', 'para', 'dahil', 'kasi', 'pero',
    'kung', 'kapag', 'habang', 'kahit', 'man', 'din', 'rin', 'daw',
    'raw', 'ba', 'po', 'opo', 'hindi', 'wala', 'walang', 'may', 'mayroon',
    'kaya', 'gusto', 'ayaw', 'dapat', 'pwede', 'puwede', 'maaari',
    'marami', 'konti', 'lahat', 'ilan', 'bawat', 'ibang', 'mismo',
    'sabi', 'sabihin', 'kailangan', 'kelangan', 'noon', 'ngayon',
    'bukas', 'kahapon', 'mamaya', 'kanina', 'matagal', 'sandali',
    'palagi', 'minsan', 'lagi', 'madalas', 'bihira', 'bago', 'luma',
    'maganda', 'pangit', 'mabuti', 'masama', 'malaki', 'maliit',
    'mahaba', 'maikli', 'matanda', 'bata', 'matamis', 'mapait',
    'maanghang', 'masarap', 'masaya', 'malungkot', 'galit', 'takot',
}

def get_final_label(row):
    """
    Logic to choose between the original LLM label or the human corrected label.
    Assumes empty/NaN in 'is_correct' means the original label is accepted.
    """
    is_correct = row.get('is_correct')
    corrected = row.get('corrected_label')
    
    # Normalize input to string for checking
    s_correct = str(is_correct).strip().lower()
    
    # Check for "no", "false", "0" indicating the original label is wrong
    if s_correct in ['no', 'false', '0', 'f']:
        # If corrected label exists, use it. Otherwise fallback to original.
        if pd.notna(corrected) and str(corrected).strip() != '':
            return str(corrected).strip()
            
    return str(row['label']).strip()

def map_to_final_tags(tag):
    """
    Maps raw tags (SYM, NUM, NE, etc.) to the required 3 classes: FIL, ENG, OTH.
    """
    tag = tag.upper().strip()
    
    # Direct mappings
    if tag in ['FIL', 'TAG', 'TL']: 
        return 'FIL'
    if tag in ['ENG', 'EN']: 
        return 'ENG'
    
    # Intra-word code switching is counted as Filipino (per specs: "naglunch")
    if tag == 'CS': 
        return 'FIL'
        
    if tag == 'NE':
        return 'OTH' # Or 'ENG' if you prefer, but OTH is safer for names like "Rizal"
        
    # Everything else (SYM, NUM, PUNCT, UNK, etc.)
    return 'OTH'

def extract_features(tokens):
    """Enhanced feature extraction for better Filipino/English discrimination"""
    feature_list = []
    padded = ['_START_', '_START2_'] + tokens + ['_END_', '_END2_']
    
    for i in range(2, len(padded) - 2):
        token = padded[i]
        prev_word = padded[i-1].lower()
        prev2_word = padded[i-2].lower()
        next_word = padded[i+1].lower()
        next2_word = padded[i+2].lower()
        word_lower = token.lower()
        
        # Character analysis
        vowels = sum(1 for c in word_lower if c in 'aeiou')
        consonants = sum(1 for c in word_lower if c.isalpha() and c not in 'aeiou')
        vowel_ratio = vowels / len(word_lower) if len(word_lower) > 0 else 0
        
        # Word list membership
        is_common_english = word_lower in COMMON_ENGLISH_WORDS
        is_common_filipino = word_lower in COMMON_FILIPINO_WORDS
        
        features = {
            # Basic features
            'word': word_lower,
            'len': len(token),
            'is_upper': token.isupper(),
            'is_title': token.istitle(),
            'vowel_ratio': vowel_ratio,
            'starts_vowel': word_lower[0] in 'aeiou' if word_lower else False,
            
            # Word list membership (STRONG signals)
            'is_common_english': is_common_english,
            'is_common_filipino': is_common_filipino,
            
            # prefixes and suffixes
            'prefix_2': word_lower[:2] if len(word_lower) >= 2 else '',
            'prefix_3': word_lower[:3] if len(word_lower) >= 3 else '',
            'suffix_2': word_lower[-2:] if len(word_lower) >= 2 else '',
            'suffix_3': word_lower[-3:] if len(word_lower) >= 3 else '',
            
            # --- FILIPINO STRONG SIGNALS ---
            'ends_ng': word_lower.endswith('ng') and not word_lower.endswith('ing'),
            'starts_mag': word_lower.startswith('mag'),
            'starts_nag': word_lower.startswith('nag'),
            'starts_pag': word_lower.startswith('pag'),
            'starts_ka': word_lower.startswith('ka'),
            'starts_pa': word_lower.startswith('pa'),
            'starts_ma': word_lower.startswith('ma'),
            'ends_han': word_lower.endswith('han'),
            'ends_an': word_lower.endswith('an') and not word_lower.endswith('tion'),
            'ends_in': word_lower.endswith('in') and not word_lower.endswith('ing'),
            'has_duplicate_syllable': len(word_lower) > 4 and word_lower[:2] == word_lower[2:4],
            
            # --- ENGLISH STRONG SIGNALS ---
            'ends_ing': word_lower.endswith('ing'),
            'ends_ed': word_lower.endswith('ed'),
            'ends_tion': word_lower.endswith('tion'),
            'ends_sion': word_lower.endswith('sion'),
            'ends_ity': word_lower.endswith('ity'),
            'ends_ment': word_lower.endswith('ment'),
            'ends_ble': word_lower.endswith('ble'),
            'ends_ness': word_lower.endswith('ness'),
            'ends_ly': word_lower.endswith('ly'),
            'ends_er': word_lower.endswith('er'),
            'ends_est': word_lower.endswith('est'),
            'ends_ful': word_lower.endswith('ful'),
            'starts_un': word_lower.startswith('un'),
            'starts_re': word_lower.startswith('re'),
            'starts_pre': word_lower.startswith('pre'),
            
            # --- CONSONANT CLUSTERS (English-heavy) ---
            'has_th': 'th' in word_lower,
            'has_ph': 'ph' in word_lower,
            'has_ct': 'ct' in word_lower,
            'has_st': 'st' in word_lower,
            'has_sh': 'sh' in word_lower,
            'has_ch': 'ch' in word_lower,
            'has_ck': 'ck' in word_lower,
            'has_wh': 'wh' in word_lower,
            
            # rate letters daw ng Filipino
            'has_f': 'f' in word_lower,
            'has_v': 'v' in word_lower,
            'has_z': 'z' in word_lower,
            'has_x': 'x' in word_lower,
            'has_c': 'c' in word_lower and 'ch' not in word_lower,
            
            # --- CONTEXT CLUES (Filipino) ---
            'prev_word': prev_word,
            'prev2_word': prev2_word,
            'next_word': next_word,
            'next2_word': next2_word,
            'prev_ang': prev_word == 'ang',
            'prev_ng': prev_word == 'ng',
            'prev_sa': prev_word == 'sa',
            'prev_mga': prev_word == 'mga',
            'prev_na': prev_word == 'na',
            'prev_ay': prev_word == 'ay',
            'next_ang': next_word == 'ang',
            'next_ng': next_word == 'ng',
            
            # bigram context (language consistency)
            'prev_is_common_eng': prev_word in COMMON_ENGLISH_WORDS,
            'prev_is_common_fil': prev_word in COMMON_FILIPINO_WORDS,
            'prev2_is_common_eng': prev2_word in COMMON_ENGLISH_WORDS,
            'prev2_is_common_fil': prev2_word in COMMON_FILIPINO_WORDS,
            'next_is_common_eng': next_word in COMMON_ENGLISH_WORDS,
            'next_is_common_fil': next_word in COMMON_FILIPINO_WORDS,
            
            # --- CONTEXT CLUES (English) ---
            'prev_the': prev_word == 'the',
            'prev_a': prev_word == 'a',
            'prev_an': prev_word == 'an',
            'prev_is': prev_word == 'is',
            'prev_are': prev_word == 'are',
            'prev_to': prev_word == 'to',
            'prev_of': prev_word == 'of',
            'prev_in': prev_word == 'in',
            'next_the': next_word == 'the',
            'next_is': next_word == 'is',
            
            # english bigrams (strong signals)
            'bigram_lets_go': prev_word == "let's" or prev_word == 'lets',
            'bigram_to_the': prev_word == 'to' and next_word == 'the',
            'bigram_machine_learning': (prev_word == 'machine' or next_word == 'machine'),
            
            # hyphenation and special chars
            'has_hyphen': '-' in token,
        }
        feature_list.append(features)
        
    return feature_list


def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except Exception:
        df = pd.read_excel(DATASET_FILE.replace('.csv', ''))

    if 'is_dirty' in df.columns:
        df = df[df['is_dirty'] != True]

    print("Processing labels...")
    df['resolved_label'] = df.apply(get_final_label, axis=1)
    
    df['target'] = df['resolved_label'].apply(map_to_final_tags)
    
    print("Extracting features...")
    all_features = []
    all_labels = []
    
    for _, group in df.groupby('sentence_id'):
        words = group['word'].astype(str).tolist()
        tags = group['target'].tolist()
        
        sentence_features = extract_features(words)
        
        all_features.extend(sentence_features)
        all_labels.extend(tags)

    X_train, X_temp, y_train, y_temp = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training on {len(X_train)} tokens...")
    
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Class weights: {class_weight_dict}")
    
    pipeline = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,          # More trees = better performance
            max_depth=20,              # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,  # Handle class imbalance
            random_state=42,
            n_jobs=-1                  # Use all CPU cores
        ))
    ])
    
    # --- Train ---
    print("Training Random Forest...")
    pipeline.fit(X_train, y_train)
    
    # --- Evaluate ---
    print("\nValidation Results:")
    y_pred_val = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred_val))
    
    print("\nTest Results:")
    y_pred_test = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    
    # --- Save ---
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()