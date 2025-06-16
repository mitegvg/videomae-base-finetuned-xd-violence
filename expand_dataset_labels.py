import pandas as pd
import os

def get_main_categories(label_str):
    """
    Extracts main non-numeric categories from a label string.
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

def expand_labels_in_file(input_csv_path, output_csv_path):
    """
    Reads a CSV file, expands multi-category labels into separate rows,
    and saves the result to a new CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path, header=None, names=['video_path', 'label'], sep=' ')
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading {input_csv_path}: {e}")
        return

    expanded_data = []
    for index, row in df.iterrows():
        video_path = row['video_path']
        original_label = row['label']
        
        main_categories = get_main_categories(original_label)
        
        if not main_categories: # Handle cases where label might be purely numeric or empty after split
            # Optionally, decide if you want to keep such records with an empty/placeholder label
            # For now, we'll skip them if no valid main categories are found.
            # Or, if the original label was simple (like "A") and get_main_categories returns it, it will be processed.
            if original_label and not original_label.isdigit(): # if original label was 'A', it's kept
                 expanded_data.append({'video_path': video_path, 'label': original_label})
            # else:
            # print(f"Skipping row with unparsed or numeric-only label: {video_path}, {original_label}")
        else:
            for category in main_categories:
                expanded_data.append({'video_path': video_path, 'label': category})

    if not expanded_data:
        print(f"No data to write for {output_csv_path} (possibly all labels were unparsable or empty).")
        return

    expanded_df = pd.DataFrame(expanded_data)
    
    try:
        expanded_df.to_csv(output_csv_path, header=False, index=False, sep=' ')
        print(f"Successfully created expanded file: {output_csv_path}")
    except Exception as e:
        print(f"Error writing {output_csv_path}: {e}")

if __name__ == "__main__":
    base_dir = "./processed_dataset/"
    
    files_to_process = {
        "train.csv": "train_expanded.csv",
        "val.csv": "val_expanded.csv",
        "test.csv": "test_expanded.csv"
    }
    
    for original_file, expanded_file in files_to_process.items():
        input_path = os.path.join(base_dir, original_file)
        output_path = os.path.join(base_dir, expanded_file)
        print(f"\nProcessing {input_path} -> {output_path}...")
        expand_labels_in_file(input_path, output_path)
