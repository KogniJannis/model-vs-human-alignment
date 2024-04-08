'''
url to load pretrained weights for ImageNet classification on DINO features
source: https://github.com/facebookresearch/dino/tree/main
note: the repo uses DeiT and ViT interchangeably
'''

dinov1_linear_urls = {
    'dino_vits16_linear':'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth',
    'dino_vits8_linear':'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth',
    'dino_vitb16_linear':'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth',
    'dino_vitb8_linear':'https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth',
    'dino_xcit_small_12_p16_linear':'https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_linearweights.pth',
    'dino_xcit_small_12_p8_linear':'https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_linearweights.pth',
    'dino_xcit_medium_24_p16_linear':'https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_linearweights.pth',
    'dino_xcit_medium_24_p8_linear':'https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_linearweights.pth',
    'dino_resnet50_linear':'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth'
}
