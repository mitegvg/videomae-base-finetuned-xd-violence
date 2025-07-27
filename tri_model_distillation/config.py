"""
Configuration classes for Tri-Model Asymmetric Distillation Framework
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from transformers import TrainingArguments


@dataclass
class TriModelConfig:
    """Configuration for the Tri-Model Asymmetric Distillation Framework"""
    
    # Model configurations
    teacher_model_name: str = "MCG-NJU/videomae-base"
    assistant_model_path: Optional[str] = None  # Path to SSV2 pretrained model
    student_model_name: str = "MCG-NJU/videomae-base"
    num_labels: int = 2  # Number of target classes
    
    # Student model size configuration
    use_tiny_student: bool = True  # Create a tiny student model
    student_num_layers: int = 4  # Number of layers in student model (vs 12 in teacher)
    student_hidden_size: int = 384  # Hidden size for student (vs 768 in teacher)
    student_num_attention_heads: int = 6  # Attention heads (vs 12 in teacher)
    
    # Distillation weights
    feature_distillation_weight: float = 1.0
    attention_distillation_weight: float = 0.5
    classification_loss_weight: float = 1.0
    
    # Temperature for distillation
    temperature: float = 4.0
    
    # Feature alignment configurations
    align_hidden_states: bool = True
    align_attention_maps: bool = True
    hidden_layers_to_align: List[int] = field(default_factory=lambda: [-1, -2, -3])  # Last 3 layers
    
    # Assistant model configurations
    assistant_feature_weight: float = 0.8
    teacher_feature_weight: float = 1.0
    
    # Training configurations
    num_frames: int = 16
    image_size: int = 224
    mask_ratio: float = 0.0  # No masking for student during distillation
    
    # Dataset configurations
    dataset_root: str = "processed_dataset"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    
    # Output configurations
    output_dir: str = "tri_model_distilled_videomae"
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    
    # Logging
    logging_steps: int = 10
    save_total_limit: int = 3
    
    def to_training_args(self, **kwargs) -> TrainingArguments:
        """Convert to TrainingArguments for HuggingFace Trainer"""
        args = {
            "output_dir": self.output_dir,
            "save_strategy": self.save_strategy,
            "eval_strategy": self.eval_strategy,
            "logging_steps": self.logging_steps,
            "save_total_limit": self.save_total_limit,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            **kwargs
        }
        return TrainingArguments(**args)


@dataclass
class SSV2ModelConfig:
    """Configuration for downloading and using SSV2 pretrained models"""
    
    # AMD MODEL_ZOO SSV2 model URLs
    ssv2_vit_s_pretrain_url: str = "https://drive.google.com/file/d/12hEVVdrFv0VNAIqAIZ0UTeicn-tm4jva/view?usp=drive_link"
    ssv2_vit_s_finetune_url: str = "https://drive.google.com/file/d/1ynDyu3K_INoZjaNLzFIaYCo6kjBOwG__/view?usp=sharing"
    ssv2_vit_b_pretrain_url: str = "https://drive.google.com/file/d/13EOb--vymBQpLNbztcN7wkdxXJiVfyXa/view?usp=sharing"
    ssv2_vit_b_finetune_url: str = "https://drive.google.com/file/d/1zc3ITIp4rR-dSelH0KaqMA9ahyju7sV2/view?usp=drive_link"
    
    # Local storage paths
    models_dir: str = "ssv2_models"
    default_backbone: str = "ViT-B"  # ViT-S or ViT-B
    use_finetuned: bool = True  # Use fine-tuned version instead of pre-trained
    num_classes: int = 174  # SSV2 has 174 classes
    
    def get_model_path(self, backbone: str = None, use_finetuned: bool = None) -> str:
        """Get local path for the specified model"""
        backbone = backbone or self.default_backbone
        use_finetuned = use_finetuned if use_finetuned is not None else self.use_finetuned
        
        model_type = "finetune" if use_finetuned else "pretrain"
        model_name = f"ssv2_vit_{backbone.lower().replace('-', '_')}_{model_type}.pth"
        
        return os.path.join(self.models_dir, model_name)
    
    def get_download_url(self, backbone: str = None, use_finetuned: bool = None) -> str:
        """Get download URL for the specified model"""
        backbone = backbone or self.default_backbone
        use_finetuned = use_finetuned if use_finetuned is not None else self.use_finetuned
        
        if backbone == "ViT-S":
            return self.ssv2_vit_s_finetune_url if use_finetuned else self.ssv2_vit_s_pretrain_url
        elif backbone == "ViT-B":
            return self.ssv2_vit_b_finetune_url if use_finetuned else self.ssv2_vit_b_pretrain_url
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'ViT-S' or 'ViT-B'")
