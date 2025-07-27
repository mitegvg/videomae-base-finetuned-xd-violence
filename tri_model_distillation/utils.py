"""
Utility functions for the Tri-Model Asymmetric Distillation Framework

This module provides helper functions for data processing, model management,
and evaluation utilities.
"""

import os
import csv
import torch
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey, Normalize, RandomShortSideScale, 
    RemoveKey, ShortSideScale, UniformTemporalSubsample
)
from torchvision.transforms import (
    Compose, Lambda, RandomCrop, RandomHorizontalFlip, Resize
)
from transformers import VideoMAEImageProcessor

logger = logging.getLogger(__name__)


def load_label_mappings(dataset_root: str, train_csv: str = "train.csv") -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load label mappings from training CSV file.
    
    Args:
        dataset_root: Root directory containing CSV files
        train_csv: Name of training CSV file
        
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    label2id = {}
    id2label = {}
    
    train_csv_path = os.path.join(dataset_root, train_csv)
    
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")
    
    with open(train_csv_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) > 1:
                label = parts[1].split("-")[0]  # Extract main class
                if label not in label2id:
                    idx = len(label2id)
                    label2id[label] = idx
                    id2label[idx] = label
    
    logger.info(f"Loaded {len(label2id)} unique labels: {list(label2id.keys())}")
    return label2id, id2label


def load_labeled_video_paths(csv_file: str, dataset_root: str, label2id: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Load video paths and labels from CSV file.
    
    Args:
        csv_file: CSV file name (e.g., "train.csv")
        dataset_root: Root directory containing CSV files
        label2id: Label to ID mapping
        
    Returns:
        List of (video_path, label_id) tuples
    """
    labeled_video_paths = []
    csv_path = os.path.join(dataset_root, csv_file)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    with open(csv_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) > 1:
                video_rel_path = parts[0]
                label_str = parts[1].split("-")[0]  # Extract main class
                
                if label_str in label2id:
                    video_full_path = os.path.join(dataset_root, video_rel_path)
                    if os.path.exists(video_full_path):
                        labeled_video_paths.append((video_full_path, label2id[label_str]))
                    else:
                        logger.warning(f"Video file not found: {video_full_path}")
                else:
                    logger.warning(f"Unknown label: {label_str}")
    
    logger.info(f"Loaded {len(labeled_video_paths)} video paths from {csv_file}")
    return labeled_video_paths


class VideoMAEDataset(Dataset):
    """
    Custom dataset for VideoMAE training with tri-model distillation.
    """
    
    def __init__(
        self,
        video_paths_and_labels: List[Tuple[str, int]],
        image_processor: VideoMAEImageProcessor,
        num_frames: int = 16,
        is_training: bool = True
    ):
        self.video_paths_and_labels = video_paths_and_labels
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.is_training = is_training
    
    def __len__(self) -> int:
        return len(self.video_paths_and_labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path, label = self.video_paths_and_labels[idx]
        
        try:
            # Load video frames
            frames = self._load_video_frames(video_path, self.num_frames)
            
            # Process frames
            inputs = self.image_processor(
                frames, 
                return_tensors="pt", 
                do_rescale=False  # Frames are already normalized
            )
            
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),  # Remove batch dimension
                "labels": torch.tensor(label, dtype=torch.long)
            }
        
        except Exception as e:
            logger.warning(f"Error loading video {video_path}: {e}")
            # Return dummy data in case of error
            dummy_frames = torch.zeros((self.num_frames, 3, 224, 224))
            return {
                "pixel_values": dummy_frames,
                "labels": torch.tensor(0, dtype=torch.long)  # Default to first class
            }
    
    def _load_video_frames(self, video_path: str, num_frames: int) -> List[np.ndarray]:
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # Use last successful frame if read fails
                if frames:
                    frame = frames[-1]
                else:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
        
        cap.release()
        
        return frames


def create_data_loaders(
    dataset_root: str,
    image_processor: VideoMAEImageProcessor,
    label2id: Dict[str, int],
    batch_size: int = 8,
    num_frames: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_root: Root directory containing CSV files and videos
        image_processor: VideoMAE image processor
        label2id: Label to ID mapping
        batch_size: Batch size for data loaders
        num_frames: Number of frames per video
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Load video paths and labels
    train_paths = load_labeled_video_paths("train.csv", dataset_root, label2id)
    val_paths = load_labeled_video_paths("val.csv", dataset_root, label2id)
    test_paths = load_labeled_video_paths("test.csv", dataset_root, label2id)
    
    # Create datasets
    train_dataset = VideoMAEDataset(
        train_paths, image_processor, num_frames, is_training=True
    )
    val_dataset = VideoMAEDataset(
        val_paths, image_processor, num_frames, is_training=False
    )
    test_dataset = VideoMAEDataset(
        test_paths, image_processor, num_frames, is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching video data.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tri_model_distillation.log')
        ]
    )


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model: torch.nn.Module, model_name: str = "Model") -> None:
    """Print model information including parameter counts."""
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n{model_name} Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")


def save_training_config(config_dict: Dict[str, Any], save_path: str) -> None:
    """Save training configuration to JSON file."""
    import json
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Training configuration saved to {save_path}")


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str]
) -> Dict[str, float]:
    """
    Evaluate model on a dataset and return metrics.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        id2label: ID to label mapping
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            # Compute loss
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    avg_loss = total_loss / num_batches
    
    # Per-class accuracy
    unique_labels = np.unique(all_labels)
    per_class_accuracy = {}
    
    for label_id in unique_labels:
        mask = np.array(all_labels) == label_id
        if np.sum(mask) > 0:
            class_accuracy = np.mean(np.array(all_predictions)[mask] == np.array(all_labels)[mask])
            label_name = id2label.get(label_id, f"class_{label_id}")
            per_class_accuracy[f"{label_name}_accuracy"] = class_accuracy
    
    metrics = {
        "accuracy": accuracy,
        "loss": avg_loss,
        **per_class_accuracy
    }
    
    return metrics


def load_xd_violence_dataset(csv_path: str, videos_dir: str, split: str) -> List[Dict[str, Any]]:
    """
    Load XD-Violence dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file containing video paths and labels
        videos_dir: Directory containing video files
        split: Dataset split name
        
    Returns:
        List of dataset samples with video paths and labels
    """
    dataset = []
    
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return dataset
    
    # XD-Violence label mapping to binary classification
    # A = Normal/Non-violent, B1/B2/B4/B5/B6 = various types of violence, G = Normal/Non-violent
    violence_labels = {'B1', 'B2', 'B4', 'B5', 'B6'}  # Violent classes
    normal_labels = {'A', 'G'}  # Non-violent classes
    
    with open(csv_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # Split by space (not comma)
            parts = line.split()
            if len(parts) >= 2:
                video_rel_path = parts[0]
                original_label = parts[1]
                
                # Map to binary classification
                if original_label in violence_labels:
                    label = 1  # Violent
                elif original_label in normal_labels:
                    label = 0  # Non-violent
                else:
                    logger.warning(f"Unknown label '{original_label}' at line {line_num}, skipping")
                    continue
                
                # Construct full video path
                # Handle videos\filename format in CSV - extract just the filename
                if video_rel_path.startswith('videos\\') or video_rel_path.startswith('videos/'):
                    filename = video_rel_path.split('\\')[-1].split('/')[-1]  # Extract filename only
                    video_path = os.path.join(videos_dir, filename)
                else:
                    video_path = os.path.join(videos_dir, video_rel_path.replace('\\', os.sep))
                
                
                if os.path.exists(video_path):
                    dataset.append({
                        'video_path': video_path,
                        'label': label,
                        'original_label': original_label,
                        'split': split
                    })
                else:
                    logger.warning(f"Video file not found: {video_path}")
            else:
                logger.warning(f"Invalid format at line {line_num}: {line}")
    
    logger.info(f"Loaded {len(dataset)} samples from {csv_path}")
    
    # Print label distribution
    violent_count = sum(1 for sample in dataset if sample['label'] == 1)
    non_violent_count = sum(1 for sample in dataset if sample['label'] == 0)
    logger.info(f"Label distribution - Violent: {violent_count}, Non-violent: {non_violent_count}")
    
    return dataset


def preprocess_video_data(dataset: List[Dict[str, Any]], image_processor, max_samples: int = None) -> List[Dict[str, torch.Tensor]]:
    """
    Memory-efficient preprocessing of video data using VideoMAE image processor.
    
    Args:
        dataset: List of dataset samples
        image_processor: VideoMAE image processor
        max_samples: Maximum number of samples to process (for testing)
        
    Returns:
        Processed dataset with pixel values and labels
    """
    import gc
    import torch
    
    processed_dataset = []
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    logger.info(f"Starting memory-efficient preprocessing of {len(dataset)} samples")
    
    # Process in smaller batches to avoid memory issues
    batch_size = 10  # Process 10 videos at a time
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_samples = dataset[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (samples {start_idx}-{end_idx})")
        
        batch_processed = []
        
        for i, sample in enumerate(batch_samples):
            try:
                video_path = sample['video_path']
                label = sample['label']
                
                logger.debug(f"Processing sample {start_idx + i + 1}: {video_path}")
                
                # Load video frames with memory management
                frames = _load_video_frames_efficient(video_path)
                
                if frames is None:
                    logger.warning(f"Skipping sample due to video loading failure: {video_path}")
                    continue
                
                # Process with image processor (this returns tensors on CPU by default)
                try:
                    inputs = image_processor(frames, return_tensors="pt")
                    
                    # Ensure tensors are on CPU to save GPU memory
                    pixel_values = inputs['pixel_values'].squeeze(0).cpu()
                    
                    batch_processed.append({
                        'pixel_values': pixel_values,
                        'labels': torch.tensor(label, dtype=torch.long)
                    })
                    
                    # Clear intermediate tensors
                    del inputs, frames
                    
                except Exception as e:
                    logger.warning(f"Image processor failed for {video_path}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed to process video {sample['video_path']}: {e}")
                continue
        
        # Add batch to main dataset
        processed_dataset.extend(batch_processed)
        
        # Memory cleanup after each batch
        del batch_processed
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 5 == 0:  # Report every 5 batches
            logger.info(f"Processed {len(processed_dataset)} videos so far, memory cleaned")
    
    logger.info(f"Successfully processed {len(processed_dataset)} videos with memory management")
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return processed_dataset


def _load_video_frames_efficient(video_path: str, num_frames: int = 16) -> Optional[List[np.ndarray]]:
    """
    Memory-efficient video frame loading.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        List of frame arrays or None if failed
    """
    import cv2
    import numpy as np
    
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return None
    
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.warning(f"No frames found in video: {video_path}")
            return None
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
        
        frames = []
        last_valid_frame = None
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to standard size (224x224)
                frame = cv2.resize(frame, (224, 224))
                
                # Normalize to [0, 1] and convert to float32
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
                last_valid_frame = frame.copy()
                
            else:
                # Use last valid frame if available, otherwise create dummy frame
                if last_valid_frame is not None:
                    frames.append(last_valid_frame.copy())
                else:
                    # Create dummy frame
                    dummy_frame = np.zeros((224, 224, 3), dtype=np.float32)
                    frames.append(dummy_frame)
        
        if len(frames) != num_frames:
            logger.warning(f"Expected {num_frames} frames, got {len(frames)} from {video_path}")
            
            # Pad with last frame if needed
            while len(frames) < num_frames:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.float32))
        
        return frames
        
    except Exception as e:
        logger.warning(f"Error loading video {video_path}: {e}")
        return None
        
    finally:
        if cap is not None:
            cap.release()


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        eval_preds: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of computed metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    predictions, labels = eval_preds
    
    # Get predicted class (argmax of logits)
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
