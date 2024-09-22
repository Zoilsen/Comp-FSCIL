# CompFSCIL

ICML 2024 paper: Compositional Few-Shot Class-Incremental Learning

## Abstract
Few-shot class-incremental learning (FSCIL) is proposed to continually learn from novel classes with only a few samples after the (pre-)training on base classes with sufficient data. However, this remains a challenge. In contrast, humans can easily recognize novel classes with a few samples. Cognitive science demonstrates that an important component of such human capability is compositional learning. This involves identifying visual primitives from learned knowledge and then composing new concepts using these transferred primitives, making incremental learning both effective and interpretable. To imitate human compositional learning, we propose a cognitive-inspired method for the FSCIL task. We define and build a compositional model based on set similarities, and then equip it with a primitive composition module and a primitive reuse module. In the primitive composition module, we propose to utilize the Centered Kernel Alignment (CKA) similarity to approximate the similarity between primitive sets, allowing the training and evaluation based on primitive compositions. In the primitive reuse module, we enhance primitive reusability by classifying inputs based on primitives replaced with the closest primitives from other classes. Experiments on three datasets validate our method, showing it outperforms current state-of-the-art methods with improved interpretability. Our code is available at https://github.com/Zoilsen/Comp-FSCIL.

## Requirements
- [PyTorch >= version 1.1](https://pytorch.org)
- tqdm

We follow [CEC](https://github.com/icoz69/CEC-CVPR2021) to use the same data index list for training. Please first follow CEC to prepare the data under the `data/` folder.

## Training scripts

CIFAR, resnet12, please specify the backbone network in models/base/Network.py, and remove the pooling in the first two residual stages.

    $ python train.py -project base -dataset cifar100 -base_mode ft_dot -new_mode ft_dot -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 150 -schedule Milestone -milestones 70 80 -temperature 1 -num_workers 8 -tag myTag -map_metric_option cka -map_metric_cls_w 2.0 -backbone_feat_cls_weight 1.0 -not_data_init -bkb_feat_pow 1.1 -map_pow 0.6 -aux_param 64 -primitive_recon_cls_weight 4.0 -gpu 0 -ft_prim_recon_tau 16.0 -ft_primitive_recon_weight 16.0


CIFAR, resnet20, please specify the backbone network in models/base/Network.py

    $ python train.py -project base -dataset cifar100 -base_mode ft_dot -new_mode ft_dot -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 150 -schedule Milestone -milestones 60 70 100 -temperature 1 -num_workers 8 -tag myTag -map_metric_option cka -map_metric_cls_w 0.5 -backbone_feat_cls_weight 1.0 -not_data_init -bkb_feat_pow 1.2 -map_pow 0.8 -aux_param 1024 -primitive_recon_cls_weight 1.0 -gpu 0


CUB200
	
    $ python train.py -project base -dataset cub200 -base_mode ft_cos -new_mode ft_cos -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 80 -schedule Milestone -milestones 15 40 60 -temperature 16 -num_workers 8 -tag myTag -map_metric_option cka -map_metric_cls_w 0.01 -backbone_feat_cls_weight 1.0 -not_data_init -bkb_feat_pow 1.2 -map_pow 0.5 -aux_param 4096 -primitive_recon_cls_weight 0.001 -gpu 0


miniImageNet, resnet12, please specify the backbone network in models/base/Network.py, and set the pooling in the first two stages as True.

    $ python train.py -project base -dataset mini_imagenet -base_mode ft_dot -new_mode ft_dot -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 200 -schedule Milestone -milestones 150 170 -temperature 1 -num_workers 8 -tag myTag -map_metric_option cka -map_metric_cls_w 2.0 -backbone_feat_cls_weight 1.0 -not_data_init -bkb_feat_pow 1.1 -map_pow 0.8 -aux_param 64 -primitive_recon_cls_weight 4.0 -gpu 0


miniImageNet, resnet18, please specify the backbone network in models/base/Network.py, and set the shape of loaded image to 224 in dataloader/miniimagenet/miniimagenet.py

    $ python train.py -project base -dataset mini_imagenet -base_mode ft_dot -new_mode ft_dot -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 150 -schedule Milestone -milestones 90 120 -temperature 1 -num_workers 8 -tag myTag -map_metric_option cka -map_metric_cls_w 4.0 -backbone_feat_cls_weight 1.0 -not_data_init -bkb_feat_pow 1.2 -map_pow 0.7 -aux_param 16 -primitive_recon_cls_weight 4.0 -gpu 0


## Acknowledgment

- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [CLOM](https://github.com/Zoilsen/CLOM)
