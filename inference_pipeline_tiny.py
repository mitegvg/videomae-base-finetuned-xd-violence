import os
import torch
from transformers import pipeline
import glob

def infer_with_pipeline():
    """
    Performs video classification using the fine-tuned VideoMAE model
    with the transformers pipeline.
    """
    model_dir = "videomae-tiny-finetuned-xd-violence"
    absolute_model_path = os.path.abspath(model_dir)
    test_videos_dir = os.path.join("test", "videos")

    if not os.path.isdir(absolute_model_path):
        print(f"ERROR: Model directory not found at {absolute_model_path}")
        print("Please ensure the 'videomae-tiny-finetuned-xd-violence' model directory exists.")
        return

    if not os.path.isdir(test_videos_dir):
        print(f"ERROR: Test videos directory not found at {test_videos_dir}")
        print("Please ensure the 'test/videos' directory exists and contains video files.")
        return

    video_files = glob.glob(os.path.join(test_videos_dir, "*.mp4")) # Add other extensions if needed
    if not video_files:
        print(f"No video files found in {test_videos_dir}")
        return

    print(f"Found {len(video_files)} videos in {test_videos_dir}")

    try:
        print(f"Loading model from: {absolute_model_path}")
        video_cls = pipeline(
            task="video-classification",
            top_k=40,
            model=absolute_model_path,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        print(f"Pipeline initialized. Using device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    print("\nStarting inference...")
    for video_path in video_files:
        print(f"\nProcessing video: {video_path}")
        try:
            raw_results = video_cls(video_path)
            print("Inference result (aggregated by main category):")
            print("Raw results:", raw_results)
            
            aggregated_scores = {}
            if raw_results:
                for res_item in raw_results:
                    original_label = res_item['label']
                    score = res_item['score']
                    main_category = original_label.split('-')[0]
                    
                    aggregated_scores[main_category] = aggregated_scores.get(main_category, 0.0) + score
            
            if aggregated_scores:
                # Sort by score descending for better readability
                sorted_aggregated_scores = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
                for label, summed_score in sorted_aggregated_scores:
                    print(f"  Main Category: {label}, Aggregated Score: {summed_score:.4f}")
            else:
                print("  No predictions returned by the pipeline.")
                
        except Exception as e:
            print(f"An error occurred during classification of {video_path}: {e}")

if __name__ == "__main__":
    infer_with_pipeline()
