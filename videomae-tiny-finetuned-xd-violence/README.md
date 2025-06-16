---
library_name: transformers
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: videomae-tiny-finetuned-xd-violence
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# videomae-tiny-finetuned-xd-violence

This model is a fine-tuned version of [](https://huggingface.co/) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2757
- Accuracy: 0.6470

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- training_steps: 1580

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.6576        | 0.25  | 395  | 1.5471          | 0.6037   |
| 1.4533        | 1.25  | 790  | 1.2815          | 0.6548   |
| 1.5216        | 2.25  | 1185 | 1.3293          | 0.6363   |
| 1.3845        | 3.25  | 1580 | 1.2757          | 0.6470   |


### Framework versions

- Transformers 4.51.3
- Pytorch 2.1.0+cu118
- Datasets 3.6.0
- Tokenizers 0.21.1
