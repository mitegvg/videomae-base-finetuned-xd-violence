"""
Tri-Model Asymmetric Distillation Trainer

This module implements the custom trainer for the tri-model distillation framework,
integrating with HuggingFace Trainer while supporting multiple models and custom loss functions.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
import numpy as np

# Handle both relative and absolute imports
try:
    from .models import TriModelDistillationFramework
    from .losses import TriModelDistillationLoss
    from .config import TriModelConfig
except ImportError:
    from models import TriModelDistillationFramework
    from losses import TriModelDistillationLoss
    from config import TriModelConfig

logger = logging.getLogger(__name__)


class TriModelDistillationTrainer(Trainer):
    """
    Custom trainer for tri-model asymmetric distillation.
    
    Extends HuggingFace Trainer to support:
    1. Multiple models (teacher, assistant, student)
    2. Custom distillation loss functions
    3. Asymmetric knowledge transfer
    """
    
    def __init__(
        self,
        framework: TriModelDistillationFramework,
        distillation_config: TriModelConfig,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        **kwargs
    ):
        
        # Initialize distillation loss
        self.distillation_loss_fn = TriModelDistillationLoss(
            feature_distillation_weight=distillation_config.feature_distillation_weight,
            attention_distillation_weight=distillation_config.attention_distillation_weight,
            classification_loss_weight=distillation_config.classification_loss_weight,
            teacher_feature_weight=distillation_config.teacher_feature_weight,
            assistant_feature_weight=distillation_config.assistant_feature_weight,
            temperature=distillation_config.temperature,
            hidden_layers_to_align=distillation_config.hidden_layers_to_align
        )
        
        self.framework = framework
        self.distillation_config = distillation_config
        
        # Use student model as the main model for Trainer
        super().__init__(
            model=framework.get_student_model(),
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute tri-model distillation loss.
        
        Args:
            model: Student model (from parent Trainer)
            inputs: Batch inputs
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (new in transformers 4.x)
            
        Returns:
            Loss tensor and optionally model outputs
        """
        
        # Extract inputs
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]
        
        # Forward pass through all models
        outputs = self.framework(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Compute distillation loss
        loss_dict = self.distillation_loss_fn(
            student_outputs=outputs['student'],
            teacher_outputs=outputs['teacher'],
            assistant_outputs=outputs['assistant'],
            labels=labels
        )
        
        total_loss = loss_dict['total_loss']
        
        # Log individual loss components
        if self.state.global_step % self.args.logging_steps == 0:
            for loss_name, loss_value in loss_dict.items():
                if loss_name != 'total_loss':
                    self.log({f"train/{loss_name}": loss_value.item()})
        
        if return_outputs:
            return total_loss, outputs['student']
        else:
            return total_loss
    
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluation loop with tri-model distillation.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self.framework.get_student_model()
        model.eval()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_feature_loss = 0.0
        total_attention_loss = 0.0
        num_samples = 0
        
        all_predictions = []
        all_labels = []
        
        for step, inputs in enumerate(eval_dataloader):
            with torch.no_grad():
                # Move inputs to device
                inputs = self._prepare_inputs(inputs)
                pixel_values = inputs["pixel_values"]
                labels = inputs["labels"]
                
                # Forward pass
                outputs = self.framework(
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                    output_attentions=True
                )
                
                # Compute losses
                loss_dict = self.distillation_loss_fn(
                    student_outputs=outputs['student'],
                    teacher_outputs=outputs['teacher'],
                    assistant_outputs=outputs['assistant'],
                    labels=labels
                )
                
                # Accumulate losses
                batch_size = labels.size(0)
                total_loss += loss_dict['total_loss'].item() * batch_size
                total_classification_loss += loss_dict['classification_loss'].item() * batch_size
                total_feature_loss += loss_dict['feature_distillation_loss'].item() * batch_size
                total_attention_loss += loss_dict['attention_distillation_loss'].item() * batch_size
                num_samples += batch_size
                
                # Collect predictions for metrics
                predictions = torch.argmax(outputs['student'].logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute average losses
        avg_loss = total_loss / num_samples
        avg_classification_loss = total_classification_loss / num_samples
        avg_feature_loss = total_feature_loss / num_samples
        avg_attention_loss = total_attention_loss / num_samples
        
        # Compute accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # FIXED: Use consistent metric naming without forward slashes
        eval_metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_classification_loss": avg_classification_loss,
            f"{metric_key_prefix}_feature_distillation_loss": avg_feature_loss,
            f"{metric_key_prefix}_attention_distillation_loss": avg_attention_loss,
            f"{metric_key_prefix}_accuracy": accuracy,  # Changed from eval/accuracy to eval_accuracy
        }
        
        # Additional metrics if compute_metrics is provided
        if self.compute_metrics is not None:
            additional_metrics = self.compute_metrics((all_predictions, all_labels))
            # Ensure consistent metric naming
            for k, v in additional_metrics.items():
                eval_metrics[f"{metric_key_prefix}_{k}"] = v
        
        return eval_metrics
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prediction step for evaluation.
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            pixel_values = inputs["pixel_values"]
            labels = inputs.get("labels")
            
            # Forward pass through framework
            outputs = self.framework(
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
                output_attentions=False  # Save memory during eval
            )
            
            student_outputs = outputs['student']
            
            # Compute loss if labels are available
            loss = None
            if labels is not None:
                loss_dict = self.distillation_loss_fn(
                    student_outputs=outputs['student'],
                    teacher_outputs=outputs['teacher'],
                    assistant_outputs=outputs['assistant'],
                    labels=labels
                )
                loss = loss_dict['total_loss']
            
            # Get predictions
            logits = student_outputs.logits
            
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the student model and framework configuration.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Save student model
        self.framework.save_student_model(output_dir)
        
        # Save framework configuration
        config_path = os.path.join(output_dir, "tri_model_config.json")
        import json
        
        config_dict = {
            "teacher_model_name": self.distillation_config.teacher_model_name,
            "student_model_name": self.distillation_config.student_model_name,
            "feature_distillation_weight": self.distillation_config.feature_distillation_weight,
            "attention_distillation_weight": self.distillation_config.attention_distillation_weight,
            "classification_loss_weight": self.distillation_config.classification_loss_weight,
            "teacher_feature_weight": self.distillation_config.teacher_feature_weight,
            "assistant_feature_weight": self.distillation_config.assistant_feature_weight,
            "temperature": self.distillation_config.temperature,
            "hidden_layers_to_align": self.distillation_config.hidden_layers_to_align,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Tri-model distillation framework saved to {output_dir}")
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Enhanced logging with distillation-specific metrics.
        Compatible with both old and new transformers versions.
        """
        # Add model information to logs periodically
        if self.state.global_step % (self.args.logging_steps * 10) == 0:
            model_info = self.framework.get_model_info()
            logs.update({
                "model/student_params": model_info['student']['trainable_parameters'],
                "model/teacher_params": model_info['teacher']['num_parameters'],
                "model/assistant_params": model_info['assistant']['num_parameters'],
            })
        
        # Call parent log method with compatible signature
        if start_time is not None:
            try:
                # Try new signature first (transformers >= 4.20)
                super().log(logs, start_time)
            except TypeError:
                # Fall back to old signature
                super().log(logs)
        else:
            super().log(logs)


def compute_video_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for video classification.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.argmax(predictions, axis=1)
    
    # Accuracy
    accuracy = np.mean(predictions == labels)
    
    # Top-1 accuracy (same as accuracy for argmax predictions)
    top1_accuracy = accuracy
    
    # Per-class accuracy
    unique_labels = np.unique(labels)
    per_class_accuracy = {}
    
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            class_accuracy = np.mean(predictions[mask] == labels[mask])
            per_class_accuracy[f"class_{label}_accuracy"] = class_accuracy
    
    metrics = {
        "accuracy": accuracy,
        "top1_accuracy": top1_accuracy,
        **per_class_accuracy
    }
    
    return metrics


class DistillationMetricsCallback:
    """Callback for logging additional distillation metrics during training."""
    
    def __init__(self, framework: TriModelDistillationFramework):
        self.framework = framework
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add distillation-specific metrics to logs."""
        if logs is not None and state.global_step % 100 == 0:
            # Add model-specific information
            model_info = self.framework.get_model_info()
            logs.update({
                "distillation/student_trainable_params": model_info['student']['trainable_parameters'],
                "distillation/total_teacher_params": model_info['teacher']['num_parameters'],
                "distillation/total_assistant_params": model_info['assistant']['num_parameters'],
            })
