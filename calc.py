import pandas as pd
from collections import Counter

def get_main_categories(label_str):
    """
    Extracts main categories from a label string.
    e.g., "B2-G-0" -> ["B2", "G"]
    "A" -> ["A"]
    "G-0-0" -> ["G"]
    """
    if pd.isna(label_str):
        return []
    parts = str(label_str).split('-')
    # Filter out parts that are purely numeric or empty
    main_categories = [part for part in parts if part and not part.isdigit()]
    return main_categories

train_df = pd.read_csv("./processed_dataset/train.csv", header=None, names=['video_path', 'label'], sep=' ')
val_df = pd.read_csv("./processed_dataset/val.csv", header=None, names=['video_path', 'label'], sep=' ')
test_df = pd.read_csv("./processed_dataset/test.csv", header=None, names=['video_path', 'label'], sep=' ')

def print_category_counts(df, df_name):
    print(f"\\nMain category counts for {df_name}:")
    all_main_categories = []
    for label in df['label']:
        all_main_categories.extend(get_main_categories(label))
    
    counts = Counter(all_main_categories)
    
    # Sort by count descending for better readability
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    for category, count in sorted_counts:
        print(f"  {category}: {count}")

print_category_counts(train_df, "train_df")
print_category_counts(val_df, "val_df")
print_category_counts(test_df, "test_df")

# Original value_counts (optional, can be removed or kept for comparison)
# print("\\nOriginal label value_counts (for reference):")
# print("Train DF original labels:")
# print(train_df['label'].value_counts())
# print("\\nValidation DF original labels:")
# print(val_df['label'].value_counts())
# print("\\nTest DF original labels:")
# print(test_df['label'].value_counts())
