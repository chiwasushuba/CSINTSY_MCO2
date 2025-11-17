import pickle
import os

# Global model variable
model_pipeline = None

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

def load_model():
    """
    Loads the trained model pipeline (Vectorizer + Classifier).
    """
    global model_pipeline
    model_path = 'pinoybot_model.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_pipeline = pickle.load(f)
    else:
        print(f"Error: {model_path} not found.")
        model_pipeline = None

def extract_features(tokens):
    """Enhanced feature extraction - MUST MATCH train_bot.py exactly!""" 
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
            
            # Morphology - prefixes and suffixes
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
            
            # Rare letters in Filipino
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
            
            # Bigram context (language consistency)
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
            
            # English bigrams (strong signals)
            'bigram_lets_go': prev_word == "let's" or prev_word == 'lets',
            'bigram_to_the': prev_word == 'to' and next_word == 'the',
            'bigram_machine_learning': (prev_word == 'machine' or next_word == 'machine'),
            
            # Hyphenation and special chars
            'has_hyphen': '-' in token,
        }
        feature_list.append(features)
        
    return feature_list

def tag_language(tokens):
    """
    Main entry point for the bot.
    Args:
        tokens: List of strings (words).
    Returns:
        List of tags (ENG, FIL, OTH).
    """
    global model_pipeline
    
    # Load model if not loaded yet
    if model_pipeline is None:
        load_model()
        
    # Fail safe if model still missing
    if model_pipeline is None:
        return ['OTH'] * len(tokens)
        
    # 1. Extract Features
    features = extract_features(tokens)
    
    # 2. Predict (Pipeline handles vectorization automatically)
    predicted_tags = model_pipeline.predict(features)
    
    return list(predicted_tags)

# --- For Testing Purposes ---
if __name__ == "__main__":
    # Sample Sentences
    test_cases = [
        ["Love", "kita", "."],
        ["Bawal", "ang", "corrupt", "dito", "."],
        ["Nag-lunch", "kami", "sa", "Jollibee", "kahapon", "."],
        ["Ang", "thesis", "namin", "ay", "about", "Machine", "Learning", "."],
        ["Let's", "go", "to", "the", "mall", "ng", "bukas", "!"]
    ]
    
    for tokens in test_cases:
        print(f"\nInput: {tokens}")
        tags = tag_language(tokens)
        print(f"Output: {tags}")