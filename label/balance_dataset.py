import pandas as pd
import nlpaug.augmenter.word as naw
import nltk
import warnings

warnings.filterwarnings('ignore')

# NLTK ke zaroori data files download karein (Synonyms ke liye)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- CONFIGURATION ---
INPUT_FILE = "categorized_articles.csv"  
OUTPUT_FILE = "balanced_news_data.csv"
TARGET_COUNT = 10000  # ✅ UPDATED: Har category ko 4000 tak le jayega (Total 20,000 rows)

print("🔄 Loading data...")
if INPUT_FILE.endswith('.csv'):
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)

# Sirf valid categories wali rows rakhein
df = df[df['category'] != 'Not predicted - missing title/content'].dropna(subset=['category'])

print(f"Original shape: {df.shape}")
print("Original distribution:\n", df['category'].value_counts())

# --- FLEXIBLE COLUMN DETECTION ---
# Yeh function automatically sahi column dhundh lega
def find_column(df, possible_names, required=True):
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in possible_names:
        found = lookup.get(name.lower())
        if found is not None:
            return found
    if required:
        raise ValueError(f"Could not find any of these columns: {possible_names}\nAvailable: {list(df.columns)}")
    return None

title_col = find_column(df, ["title", "title_clean", "headline"])
content_col = find_column(df, ["content", "content_clean", "text", "article_content", "description"])

print(f"\n✅ Detected columns -> Title: '{title_col}', Content: '{content_col}'")

# Augmenters initialize karein
synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_max=0.3)
swap_aug = naw.RandomWordAug(action='swap', aug_max=0.2)

def augment_text(text):
    """Text ko augment kar ke naya sentence banata hai"""
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 10:
        return text
    try:
        aug_result = synonym_aug.augment(text)
        augmented = aug_result[0] if isinstance(aug_result, list) else aug_result
        
        if augmented == text:
            aug_result = swap_aug.augment(text)
            augmented = aug_result[0] if isinstance(aug_result, list) else aug_result
            
        return augmented
    except Exception:
        return text

balanced_dfs = []

print("\n⚖️ Balancing process shuru...")
for category, group in df.groupby('category'):
    current_count = len(group)
    print(f"\nProcessing '{category}': {current_count} rows")
    
    if current_count >= TARGET_COUNT:
        print(f"  -> Undersampling to {TARGET_COUNT}...")
        balanced_group = group.sample(n=TARGET_COUNT, random_state=42)
        balanced_dfs.append(balanced_group)
    else:
        print(f"  -> Augmenting to {TARGET_COUNT} (Yeh thora time le sakta hai)...")
        augmented_group = group.copy()
        original_group = group.copy()
        
        attempts = 0
        max_attempts = 50
        
        while len(augmented_group) < TARGET_COUNT and attempts < max_attempts:
            n_to_generate = min(len(original_group), TARGET_COUNT - len(augmented_group))
            samples_to_aug = original_group.sample(n=n_to_generate, replace=True).copy()
            
            # Yahan ab dynamic column names use ho rahe hain
            samples_to_aug[content_col] = samples_to_aug[content_col].apply(augment_text)
            samples_to_aug[title_col] = samples_to_aug[title_col].apply(augment_text)
            
            augmented_group = pd.concat([augmented_group, samples_to_aug], ignore_index=True)
            attempts += 1
            
        balanced_dfs.append(augmented_group.sample(n=TARGET_COUNT, random_state=42))

final_df = pd.concat(balanced_dfs, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n✅ Balancing Complete!")
print("Final distribution:\n", final_df['category'].value_counts())
print(f"Total balanced rows: {len(final_df)}")

final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n💾 Balanced dataset save ho gaya: {OUTPUT_FILE}")