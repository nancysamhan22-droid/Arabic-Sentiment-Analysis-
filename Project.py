import pandas as pd
import numpy as np
import re
import warnings
import pickle
import emoji
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')

print("=" * 80)
print("ARABIC SENTIMENT ANALYSIS - FAST VERSION ⚡")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] LOADING DATA")
print("-" * 80)

# ⚠️ IMPORTANT: Update this path to your actual dataset file!
dataset_path = r"C:\Users\hp\OneDrive\Desktop\AiProjs\Ai2\dataset.txt"

try:
    print(f"Attempting to load dataset from: {dataset_path}")
    
    # Method 1: Try standard pandas read
    try:
        df = pd.read_csv(dataset_path, 
                        sep='\t', 
                        header=None, 
                        names=['tweet', 'sentiment'], 
                        encoding='utf-8',
                        dtype=str,
                        engine='python',
                        on_bad_lines='skip',
                        quoting=3)  # QUOTE_NONE
        
        initial_count = len(df)
        print(f"✅ Initial load: {initial_count} samples")
        
    except Exception as e:
        print(f"⚠️  Standard loading failed: {str(e)}")
        df = pd.DataFrame(columns=['tweet', 'sentiment'])
        initial_count = 0
    
    # Method 2: Check for skipped lines and manual recovery
    print("\n🔍 Verifying all lines were loaded...")
    
    with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"   Total lines in file: {total_lines}")
    print(f"   Lines loaded by pandas: {initial_count}")
    
    skipped = total_lines - initial_count
    
    if skipped > 0:
        print(f"\n⚠️  Found {skipped} problematic lines - attempting manual recovery...")
        
        # Manual line-by-line parsing
        tweets = []
        sentiments = []
        failed_lines = []
        
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    
                    if not line:  # Skip empty lines
                        continue
                    
                    # Split by tab - handle multiple tabs
                    parts = line.split('\t')
                    
                    if len(parts) >= 2:
                        # Everything except last part is the tweet
                        tweet = '\t'.join(parts[:-1]).strip()
                        sentiment = parts[-1].strip()
                        
                        if tweet and sentiment:  # Only add if both exist
                            tweets.append(tweet)
                            sentiments.append(sentiment)
                    else:
                        failed_lines.append((line_num, "Only one column"))
                        
                except Exception as e:
                    failed_lines.append((line_num, str(e)))
        
        # Create new dataframe from manual parsing
        df = pd.DataFrame({
            'tweet': tweets,
            'sentiment': sentiments
        })
        
        print(f"\n✅ Manual parsing complete!")
        print(f"   Successfully loaded: {len(df)} rows")
        print(f"   Failed to parse: {len(failed_lines)} rows")
        
        if len(failed_lines) > 0 and len(failed_lines) <= 10:
            print(f"\n   Failed lines details:")
            for line_num, error in failed_lines[:10]:
                print(f"      Line {line_num}: {error}")
    
    else:
        print("✅ All lines loaded successfully!")
    
    # Clean and normalize sentiment labels
    print("\n📊 Normalizing sentiment labels...")
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.upper()
    df['sentiment'] = df['sentiment'].replace('NEUTRAL', 'OBJ')
    
    # Remove empty tweets
    print("   Removing empty tweets...")
    before = len(df)
    df = df[df['tweet'].astype(str).str.len() > 0].reset_index(drop=True)
    after = len(df)
    
    if before != after:
        print(f"   Removed {before - after} empty tweets")
    
    print(f"\n✅ Final dataset: {len(df)} samples")
    print(f"   Class distribution: {dict(df['sentiment'].value_counts())}")
    
    # Validate we have data
    if len(df) == 0:
        print("❌ Error: No valid data loaded!")
        exit(1)
    
except FileNotFoundError:
    print(f"❌ Error: '{dataset_path}' not found!")
    print("\n💡 Please update the 'dataset_path' variable with your actual dataset file path.")
    print("   Example: dataset_path = r'C:\\Users\\hp\\OneDrive\\Desktop\\AiProjs\\Ai2\\YOUR_FILE.tsv'")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error loading dataset: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================
print("\n[STEP 2] TEXT PREPROCESSING")
print("-" * 80)

def remove_diacritics(text):
    """Remove Arabic diacritics"""
    return re.sub(r'[\u064B-\u065F\u0670]', '', text)

def normalize_arabic(text):
    """Normalize Arabic characters"""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return text

def clean_text(text):
    """Complete text cleaning pipeline"""
    if not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # Remove hashtag symbol but keep word
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Reduce repeated chars
    text = re.sub(r"[a-zA-Z0-9]+", "", text)  # Remove English/numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    return " ".join(text.split())

print("Processing tweets...")
df['processed_text'] = df['tweet'].astype(str).apply(clean_text)
before = len(df)
df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
after = len(df)

if before != after:
    print(f"   Removed {before - after} tweets with no processable text")

print(f"✅ Preprocessed: {len(df)} samples remaining")

# ============================================================================
# STEP 3: COMPACT FEATURES (12 handcrafted features)
# ============================================================================
print("\n[STEP 3] EXTRACTING FEATURES")
print("-" * 80)

POSITIVE_WORDS = {
    'جميل','رائع','ممتاز','حلو','رهيب','احب','سعيد','فرح','نجاح','جيد',
    'تمام','روعه','حب','افضل','مبهر','عظيم','ممتن','سرور','بهجه'
}

NEGATIVE_WORDS = {
    'سيء','فظيع','غضب','حزن','كره','فشل','مشكله','سخيف','مقرف',
    'تعاسه','بشع','كريه','مزعج','ضعيف','قبيح','خيبه','يأس'
}

POSITIVE_EMOJIS = {'😊','😃','😄','❤️','💕','👍','🎉','😍','🥰','💖','✨','🌟','💯','🙌','😁'}
NEGATIVE_EMOJIS = {'😢','😭','😞','😔','💔','😡','😠','😤','🤬','😩','😫','👎','💀','😱','🙁'}
NEGATIONS = {"ما","لا","لم","لن","ليس","مش","ولا","غير"}

def extract_emojis(text):
    """Extract all emojis from text"""
    return [ch for ch in text if ch in emoji.EMOJI_DATA]

def extract_features(row):
    """Extract handcrafted features from tweet"""
    text = str(row['tweet'])
    processed = str(row['processed_text'])
    words = processed.split()
    
    emojis = extract_emojis(text)
    n_words = len(words)
    
    pos_word_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_word_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    pos_emoji_count = sum(1 for e in emojis if e in POSITIVE_EMOJIS)
    neg_emoji_count = sum(1 for e in emojis if e in NEGATIVE_EMOJIS)
    
    return {
        'pos_emoji': pos_emoji_count,
        'neg_emoji': neg_emoji_count,
        'pos_word': pos_word_count,
        'neg_word': neg_word_count,
        'negation': sum(1 for w in words if w in NEGATIONS),
        'exclaim': text.count('!'),
        'question': text.count('?') + text.count('؟'),
        'char_count': len(text),
        'word_count': n_words,
        'char_repeat': len(re.findall(r'(.)\1{2,}', text)),
        'avg_word_len': sum(len(w) for w in words) / max(n_words, 1),
        'sentiment_score': (pos_word_count + pos_emoji_count) - (neg_word_count + neg_emoji_count)
    }

print("Extracting handcrafted features...")
handcrafted_df = df.apply(extract_features, axis=1, result_type='expand')
print(f"✅ Extracted {len(handcrafted_df.columns)} handcrafted features")

# ============================================================================
# STEP 4: AraBERT EMBEDDINGS (FASTER with batch processing)
# ============================================================================
print("\n[STEP 4] EXTRACTING AraBERT EMBEDDINGS")
print("-" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

try:
    print("Loading AraBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2").to(device)
    model.eval()
    print("✅ Model loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading AraBERT: {str(e)}")
    print("💡 Make sure you have internet connection and transformers library installed")
    print("   Run: pip install transformers torch")
    exit(1)

def get_embeddings_batch(texts, batch_size=32):
    """Process texts in batches for speed"""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch = texts[i:i+batch_size]
        
        inputs = tokenizer(batch, return_tensors="pt", max_length=128,
                          truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

print("Processing tweets in batches...")
embeddings_array = get_embeddings_batch(df['processed_text'].tolist(), batch_size=32)
print(f"✅ AraBERT embeddings: {embeddings_array.shape}")

# Free up memory
del model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# STEP 5: COMBINE & SPLIT
# ============================================================================
print("\n[STEP 5] COMBINING & SPLITTING DATA")
print("-" * 80)

# Combine features
X = np.hstack([handcrafted_df.values, embeddings_array])
y = df['sentiment'].values
print(f"Total features: {X.shape[1]} ({handcrafted_df.shape[1]} handcrafted + {embeddings_array.shape[1]} AraBERT)")

# Split data: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Apply SMOTE to handle class imbalance
print("\nApplying SMOTE...")
smote = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_train_smote)} samples")
print(f"New distribution: {dict(pd.Series(y_train_smote).value_counts())}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✅ Scaling complete")

# ============================================================================
# STEP 6: TRAIN DECISION TREE
# ============================================================================
print("\n[STEP 6] TRAINING DECISION TREE ⚡")
print("-" * 80)

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    {
        'max_depth': [25, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    },
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

print("Training with GridSearchCV...")
dt_grid.fit(X_train_scaled, y_train_smote)
dt_model = dt_grid.best_estimator_

y_val_pred_dt = dt_model.predict(X_val_scaled)
y_test_pred_dt = dt_model.predict(X_test_scaled)

val_acc_dt = accuracy_score(y_val, y_val_pred_dt)
test_acc_dt = accuracy_score(y_test, y_test_pred_dt)

print(f"\n✅ Best params: {dt_grid.best_params_}")
print(f"   Validation Accuracy: {val_acc_dt:.4f}")
print(f"   Test Accuracy: {test_acc_dt:.4f}")

with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
print("   Saved: decision_tree_model.pkl")

# ============================================================================
# STEP 7: TRAIN RANDOM FOREST
# ============================================================================
print("\n[STEP 7] TRAINING RANDOM FOREST ⚡")
print("-" * 80)

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    {
        'n_estimators': [200, 300],
        'max_depth': [25, 30],
        'min_samples_split': [5, 10]
    },
    cv=3,
    scoring='f1_weighted',
    n_jobs=1,
    verbose=1
)

print("Training with GridSearchCV...")
rf_grid.fit(X_train_scaled, y_train_smote)
rf_model = rf_grid.best_estimator_

y_val_pred_rf = rf_model.predict(X_val_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)

val_acc_rf = accuracy_score(y_val, y_val_pred_rf)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)

print(f"\n✅ Best params: {rf_grid.best_params_}")
print(f"   Validation Accuracy: {val_acc_rf:.4f}")
print(f"   Test Accuracy: {test_acc_rf:.4f}")

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("   Saved: random_forest_model.pkl")

# ============================================================================
# STEP 8: TRAIN NAIVE BAYES
# ============================================================================
print("\n[STEP 8] TRAINING NAIVE BAYES ⚡")
print("-" * 80)

nb_grid = GridSearchCV(
    GaussianNB(),
    {'var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3]},
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

print("Training with GridSearchCV...")
nb_grid.fit(X_train_scaled, y_train_smote)
nb_model = nb_grid.best_estimator_

y_val_pred_nb = nb_model.predict(X_val_scaled)
y_test_pred_nb = nb_model.predict(X_test_scaled)

val_acc_nb = accuracy_score(y_val, y_val_pred_nb)
test_acc_nb = accuracy_score(y_test, y_test_pred_nb)

print(f"\n✅ Best params: {nb_grid.best_params_}")
print(f"   Validation Accuracy: {val_acc_nb:.4f}")
print(f"   Test Accuracy: {test_acc_nb:.4f}")

with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
print("   Saved: naive_bayes_model.pkl")

# ============================================================================
# STEP 9: TRAIN NEURAL NETWORK (OPTIONAL)
# ============================================================================
print("\n[STEP 9] TRAINING NEURAL NETWORK ⚡")
print("-" * 80)

nn_trained = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_smote)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Build model
    nn_model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    nn_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training neural network...")
    history = nn_model.fit(
        X_train_scaled, y_train_encoded,
        validation_data=(X_val_scaled, y_val_encoded),
        epochs=30,
        batch_size=64,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ],
        verbose=1
    )
    
    # Evaluate
    y_test_pred_nn = np.argmax(nn_model.predict(X_test_scaled, verbose=0), axis=1)
    y_test_pred_nn_labels = label_encoder.inverse_transform(y_test_pred_nn)
    test_acc_nn = accuracy_score(y_test, y_test_pred_nn_labels)
    
    print(f"\n✅ Test Accuracy: {test_acc_nn:.4f}")
    print(f"   Training epochs: {len(history.history['loss'])}")
    
    nn_model.save('neural_network_model.h5')
    with open('label_encoder_nn.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("   Saved: neural_network_model.h5, label_encoder_nn.pkl")
    
    nn_trained = True
    
except ImportError:
    print("⚠️  TensorFlow not installed - skipping Neural Network")
    print("   Install with: pip install tensorflow")
except Exception as e:
    print(f"⚠️  Neural Network training failed: {str(e)[:100]}")
    print("   Continuing with 3 models only...")

# ============================================================================
# STEP 10: RESULTS & VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

models_data = [
    ('Decision Tree', y_test_pred_dt, test_acc_dt),
    ('Random Forest', y_test_pred_rf, test_acc_rf),
    ('Naive Bayes', y_test_pred_nb, test_acc_nb)
]

if nn_trained:
    models_data.append(('Neural Network', y_test_pred_nn_labels, test_acc_nn))

# Print detailed results
for name, pred, acc in models_data:
    print(f"\n{'='*80}")
    print(f"{name} - Test Accuracy: {acc:.4f}")
    print(f"{'='*80}")
    print(classification_report(y_test, pred, zero_division=0))

# Create comparison DataFrame
results_df = pd.DataFrame({
    'Model': [m[0] for m in models_data],
    'Test_Accuracy': [m[2] for m in models_data],
    'Test_F1': [f1_score(y_test, m[1], average='weighted') for m in models_data]
})

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print(results_df.to_string(index=False))

best_idx = results_df['Test_Accuracy'].idxmax()
print(f"\n🏆 Best Model: {results_df.loc[best_idx, 'Model']}")
print(f"   Accuracy: {results_df.loc[best_idx, 'Test_Accuracy']:.4f}")
print(f"   F1-Score: {results_df.loc[best_idx, 'Test_F1']:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n[VISUALIZATION] Creating plots...")
print("-" * 80)

# 1. Confusion Matrices
n_models = len(models_data)
fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
if n_models == 1:
    axes = [axes]

for idx, (name, pred, _) in enumerate(models_data):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(np.unique(y)), 
                yticklabels=sorted(np.unique(y)), 
                ax=axes[idx])
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, pred):.3f}')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrices.png")

# 2. Performance Comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['Test_Accuracy'], width, 
               label='Accuracy', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, results_df['Test_F1'], width, 
               label='F1-Score', alpha=0.8, color='coral')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: performance_comparison.png")

plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[SAVING] Exporting results...")
print("-" * 80)

results_df.to_csv('final_results.csv', index=False)
print("✅ Saved: final_results.csv")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: scaler.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETE!")
print("=" * 80)
print(f"""
📊 Summary:
   • Dataset: {len(df)} samples
   • Features: {X.shape[1]} ({handcrafted_df.shape[1]} handcrafted + {embeddings_array.shape[1]} AraBERT)
   • Models trained: {len(models_data)} ({', '.join([m[0] for m in models_data])})
   • Best model: {results_df.loc[best_idx, 'Model']}
   • Best accuracy: {results_df.loc[best_idx, 'Test_Accuracy']:.4f}
   • Best F1-score: {results_df.loc[best_idx, 'Test_F1']:.4f}
""")
print("=" * 80)
print("\n🎉 All done! Check the generated files for results.")
print("=" * 80)
