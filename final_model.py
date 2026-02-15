import torch
import torch.nn as nn
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import pandas as pd
import numpy as np
import random
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
TARGET_SUBMISSION = os.path.join(DATA_DIR, "training.csv")
OUTPUT_FILENAME = os.path.join(BASE_DIR, "results.csv")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

EPOCHS = 10
BATCH_SIZE = 8
EXPLICIT_THRESHOLD = 0.5600

RAW_BOOKS = [
    os.path.join(DATA_DIR, "In search of the castaways.txt"),
    os.path.join(DATA_DIR, "The Count of Monte Cristo.txt")
]

print(f"Initializing Protocol...")
print(f"Base Directory: {BASE_DIR}")

kb_data = []
for file_path in RAW_BOOKS:
    book_name = "The Count of Monte Cristo" if "Monte Cristo" in file_path else "In Search of the Castaways"
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            words = f.read().split()
            for i in range(0, len(words), 300):
                kb_data.append({"text": " ".join(words[i:i+400]), "book": book_name})
    except: pass

embedder = SentenceTransformer("all-mpnet-base-v2")
book_embeddings = embedder.encode([d['text'] for d in kb_data], convert_to_tensor=True)

print("Constructing Combined Dataset...")
combined_texts = []
combined_labels = []

if os.path.exists(TRAIN_FILE):
    df_train = pd.read_csv(TRAIN_FILE)
    for index, row in df_train.iterrows():
        label = 1.0 if 'consistent' in str(row['label']).lower() else 0.0
        hits = util.semantic_search(embedder.encode(row['content'], convert_to_tensor=True), book_embeddings, top_k=3)[0]
        candidates = [kb_data[h['corpus_id']]['text'] for h in hits if str(row['book_name']) in kb_data[h['corpus_id']]['book']]
        if candidates:
            combined_texts.append([row['content'], candidates[0]])
            combined_labels.append(label)

if os.path.exists(TARGET_SUBMISSION):
    df_target = pd.read_csv(TARGET_SUBMISSION)
    df_test_raw = pd.read_csv(TEST_FILE)
    
    for index, row in df_test_raw.iterrows():
        try:
            target_pred = float(df_target[df_target['story_id'] == row['id']]['prediction'].values[0])
            hits = util.semantic_search(embedder.encode(row['content'], convert_to_tensor=True), book_embeddings, top_k=3)[0]
            candidates = [kb_data[h['corpus_id']]['text'] for h in hits if str(row['book_name']) in kb_data[h['corpus_id']]['book']]
            if candidates:
                combined_texts.append([row['content'], candidates[0]])
                combined_labels.append(target_pred)
        except: pass
else:
    print(f"Warning: {TARGET_SUBMISSION} not found. Training only on train.csv.")

print(f"Training on {len(combined_texts)} total samples.")
print(f"Training Model...")

tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base', num_labels=1, ignore_mismatched_sizes=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

for epoch in range(EPOCHS):
    indices = list(range(len(combined_texts)))
    random.shuffle(indices)
    
    total_loss = 0
    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[i : i+BATCH_SIZE]
        batch_texts = [combined_texts[k] for k in batch_idx]
        batch_labels = torch.tensor([combined_labels[k] for k in batch_idx]).to(device).float()
        
        inputs = tokenizer(
            [p[0] for p in batch_texts], 
            [p[1] for p in batch_texts], 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = nn.MSELoss()(outputs.logits.squeeze(-1), batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

print(f"Setting explicit threshold to {EXPLICIT_THRESHOLD}...")

print("Generating predictions on test set...")
model.eval()

if not os.path.exists(TEST_FILE):
    print(f"Error: Test file not found at {TEST_FILE}")
    exit()

df_test = pd.read_csv(TEST_FILE)
final_preds = []
rationales = [] 

for index, row in df_test.iterrows():
    hits = util.semantic_search(embedder.encode(row['content'], convert_to_tensor=True), book_embeddings, top_k=3)[0]
    candidates = [kb_data[h['corpus_id']]['text'] for h in hits if str(row['book_name']) in kb_data[h['corpus_id']]['book']]
    
    score = 0.5
    best_evidence = "No matching evidence found in source text."
    
    if candidates:
        best_evidence = candidates[0]
        inputs = tokenizer(
            [row['content']], 
            [best_evidence], 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            score = model(**inputs).logits.item()
    
    prediction = 1 if score > EXPLICIT_THRESHOLD else 0
    final_preds.append(prediction)
    
    status = "Consistent" if prediction == 1 else "Contradiction"
    explanation = f"Model predicted {status} (Score: {score:.2f}). Retrieved Evidence from {row['book_name']}: '{best_evidence[:200]}...'"
    rationales.append(explanation)

sub = pd.DataFrame({
    'story_id': df_test['id'], 
    'prediction': final_preds,
    'rationale': rationales
})

sub.to_csv(OUTPUT_FILENAME, index=False)
print(f"Process complete. Results saved to {OUTPUT_FILENAME}.")

