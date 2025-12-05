import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from train_bot import extract_features  

MODEL_FILE = "pinoybot_model.pkl"
INPUT_TEXT_FILE = "test_data.txt"
TRUE_LABEL_FILE = "test_labels.txt"

def load_sentences():
    with open(INPUT_TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = [s.strip() for s in text.split("|") if s.strip()]
    return sentences

def load_true_labels():
    labels = []
    with open(TRUE_LABEL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            labels.extend(line.strip().split("|"))
    return labels

def tokenize_sentence(sentence):
    return sentence.split()

def main():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    sentences = load_sentences()
    true_labels = load_true_labels()

    words = []
    predicted_labels = []

    all_feature_dicts = []

    for sent in sentences:
        tokens = tokenize_sentence(sent)
        words.extend(tokens)

        features = extract_features(tokens)
        all_feature_dicts.extend(features)

    if len(words) != len(true_labels):
        print("ERROR: label count does not match word count!")
        print(f"Words: {len(words)}, Labels: {len(true_labels)}")
        return

    predicted_labels = model.predict(all_feature_dicts)

    # Get classification report as dict
    report = classification_report(true_labels, predicted_labels, digits=4, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=["FIL", "ENG", "OTH"])
    
    print("\n=== METRICS FOR EACH CATEGORY ===\n")
    
    categories = ["FIL", "ENG", "OTH"]
    for i, category in enumerate(categories):
        print(f"--- {category} ---")
        
        # Get TP, FP, FN from confusion matrix
        # cm[i, i] = True Positives
        # sum of column i (excluding diagonal) = False Positives
        # sum of row i (excluding diagonal) = False Negatives
        
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP  # All predicted as this class minus TP
        FN = cm[i, :].sum() - TP  # All actually this class minus TP
        
        # Get metrics from report
        precision = report[category]['precision']
        recall = report[category]['recall']
        f1 = report[category]['f1-score']
        support = report[category]['support']
        
        print(f"TP (True Positives):  {TP}")
        print(f"FP (False Positives): {FP}")
        print(f"FN (False Negatives): {FN}")
        print(f"Support (Actual):     {int(support)}")
        print()
        print(f"Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {TP} / {TP + FP} = {precision:.4f}")
        print(f"Recall    = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {TP} / {TP + FN} = {recall:.4f}")
        print(f"F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)")
        print(f"          = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
        print(f"          = 2 * {precision * recall:.4f} / {precision + recall:.4f}")
        print(f"          = {2 * precision * recall:.4f} / {precision + recall:.4f} = {f1:.4f}")
        print()
    
    # Overall metrics
    print("--- OVERALL ---")
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall:    {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1-Score:  {report['macro avg']['f1-score']:.4f}")
    print(f"\nTotal samples: {len(true_labels)}")

if __name__ == "__main__":
    main()
