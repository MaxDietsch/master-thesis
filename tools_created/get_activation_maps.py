
import os 
import torch

phase = 'phase3'
model = 'swin'
config_file = 'swin_sgd_decr.py'
schedule = 'lr_decr'
algo = 'ros25_aug_pretrained_focal2'
epoch = '100'
CFG = f'../config/phase1/{config_file}'
CHECKPOINT = f'../work_dirs/{phase}/{model}/{algo}/{schedule}/epoch_{epoch}.pth'
DEVICE = torch.device('cuda')
METHOD = 'EigenCAM'
AUG_SMOOTH = False
EIGEN_SMOOTH = False 
PATH = '../../utils/esophagitis'
VIT_LIKE = True
TARGET_LAYERS = []


# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import pkg_resources
from functools import partial
from pathlib import Path
import mmcv
import numpy as np
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.config import Config, DictAction
from mmengine.dataset import default_collate
from mmengine.utils import to_2tuple
from mmengine.utils.dl_utils import is_norm

from mmpretrain import digit_version
from mmpretrain.apis import get_model
from mmpretrain.registry import TRANSFORMS

try:
    import pytorch_grad_cam as cam
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# Alias name
METHOD_MAP = {
    'gradcam++': cam.GradCAMPlusPlus,
}
METHOD_MAP.update({
    cam_class.__name__.lower(): cam_class
    for cam_class in cam.base_cam.BaseCAM.__subclasses__()
})

def reshape_transform(tensor, model):
    """Build reshape_transform for `cam.activations_and_grads`, which is
    necessary for ViT-like networks."""
    # ViT_based_Transformers have an additional clstoken in features
    if tensor.ndim == 4:
        # For (B, C, H, W)
        return tensor
    elif tensor.ndim == 3:
        if not VIT_LIKE:
            raise ValueError(f"The tensor shape is {tensor.shape}, if it's a "
                             'vit-like backbone, please specify `--vit-like`.')
        # For (B, L, C)
        num_extra_tokens = getattr(
            model.backbone, 'num_extra_tokens', 1)

        tensor = tensor[:, num_extra_tokens:, :]
        # get heat_map_height and heat_map_width, preset input is a square
        heat_map_area = tensor.size()[1]
        height, width = to_2tuple(int(math.sqrt(heat_map_area)))
        assert height * height == heat_map_area, \
            (f"The input feature's length ({heat_map_area+num_extra_tokens}) "
             f'minus num-extra-tokens ({num_extra_tokens}) is {heat_map_area},'
             ' which is not a perfect square number. Please check if you used '
             'a wrong num-extra-tokens.')
        # (B, L, C) -> (B, H, W, C)
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        # (B, H, W, C) -> (B, C, H, W)
        result = result.permute(0, 3, 1, 2)
        return result
    else:
        raise ValueError(f'Unsupported tensor shape {tensor.shape}.')


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with
    mmpretrain, here we modify the ActivationsAndGradients object."""
    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers, use_cuda=use_cuda)
    # Release the original hooks in ActivationsAndGradients to use
    # ActivationsAndGradients.
    cam.activations_and_grads.release()
    cam.activations_and_grads = ActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model layer from given str."""
    for name, layer in model.named_modules():
        if name == layer_str:
            return layer
    raise AttributeError(
        f'Cannot get the layer "{layer_str}". Please choose from: \n' +
        '\n'.join(name for name, _ in model.named_modules()))


def show_cam_grad(grayscale_cam, src_img, title, out_path=None):
    """fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img) / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)

    if out_path:
        mmcv.imwrite(visualization_img, str(out_path))
    else:
        mmcv.imshow(visualization_img, win_name=title)


def get_default_target_layers(model, vit_like):
    """get default target layers from given model, here choose nrom type layer
    as default target layer."""
    norm_layers = [
        (name, layer)
        for name, layer in model.backbone.named_modules(prefix='backbone')
        if is_norm(layer)
    ]
    if vit_like:
        # For ViT models, the final classification is done on the class token.
        # And the patch tokens and class tokens won't interact each other after
        # the final attention layer. Therefore, we need to choose the norm
        # layer before the last attention layer.
        num_extra_tokens = getattr(model.backbone, 'num_extra_tokens', 1)

        # models like swin have no attr 'out_type', set out_type to avg_featmap
        out_type = getattr(model.backbone, 'out_type', 'avg_featmap')
        if out_type == 'cls_token' or num_extra_tokens > 0:
            # Assume the backbone feature is class token.
            name, layer = norm_layers[-3]
            print('Automatically choose the last norm layer before the '
                  f'final attention block "{name}" as the target layer.')
            return [layer]

    # For CNN models, use the last norm layer as the target-layer
    name, layer = norm_layers[-1]
    print('Automatically choose the last norm layer '
          f'"{name}" as the target layer.')
    return [layer]


def main():
        
    
    # build the model from a config file and a checkpoint file
    model: nn.Module = get_model(CFG, CHECKPOINT, device=DEVICE)

    

    # build target layers
    
    if TARGET_LAYERS:
        target_layers = [
            get_layer(layer, model) for layer in TARGET_LAYERS
        ]
    else:
        target_layers = get_default_target_layers(model, VIT_LIKE)

    # init a cam grad calculator
    use_cuda = True
    cam = init_cam(METHOD, model, target_layers, use_cuda, partial(reshape_transform, model=model))

    # warp the target_category with ClassifierOutputTarget in grad_cam>=1.3.7,
    # to fix the bug in #654.
    targets = None
    """
    if args.target_category:
        grad_cam_v = pkg_resources.get_distribution('grad_cam').version
        if digit_version(grad_cam_v) >= digit_version('1.3.7'):
            from pytorch_grad_cam.utils.model_targets import \
                ClassifierOutputTarget
            targets = [ClassifierOutputTarget(c) for c in args.target_category]
        else:
            targets = args.target_category
    """

    # calculate cam grads and show|save the visualization image
    cfg = Config.fromfile(CFG)
    for path in os.listdir(PATH):

        save_path = os.path.join('../../utils/cams/', path)
        file_path = os.path.join(PATH, path)
        if os.path.isfile(file_path):

            transforms = Compose([TRANSFORMS.build(t) for t in cfg.test_dataloader.dataset.pipeline])
            data = transforms({'img_path': file_path})
            src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)
            data = model.data_preprocessor(default_collate([data]), False)

        
            grayscale_cam = cam(
                data['inputs'],
                targets,
                eigen_smooth=EIGEN_SMOOTH,
                aug_smooth=AUG_SMOOTH)
            show_cam_grad(
                grayscale_cam, src_img, title=METHOD, out_path=save_path)


if __name__ == '__main__':
    main()
