"""
Tri-Model Asymmetric Distillation Framework

This package implements the Tri-model Asymmetric Distillation Framework 
based on the research paper for efficient domain-specific video classification.

Components:
- Teacher Model: VideoMAE base pretrained model (frozen)
- Assistant Model: SSV2 pretrained model from AMD MODEL_ZOO (frozen)
- Student Model: Target model for fine-tuning

The framework uses feature distillation to transfer knowledge from both
teacher and assistant models to improve the student model's performance
on domain-specific tasks.
"""

from .models import (
    TriModelDistillationFramework,
    TriModelOrchestrator, 
    load_teacher_model,
    load_assistant_model,
    load_student_model
)
from .trainer import TriModelDistillationTrainer
from .config import TriModelConfig, SSV2ModelConfig
from .losses import TriModelDistillationLoss
from .utils import (
    load_xd_violence_dataset,
    preprocess_video_data,
    compute_metrics
)

__version__ = "1.0.0"
__all__ = [
    "TriModelDistillationFramework",
    "TriModelOrchestrator",
    "TriModelDistillationTrainer", 
    "TriModelConfig",
    "SSV2ModelConfig",
    "TriModelDistillationLoss",
    "load_teacher_model",
    "load_assistant_model", 
    "load_student_model",
    "load_xd_violence_dataset",
    "preprocess_video_data",
    "compute_metrics"
]
