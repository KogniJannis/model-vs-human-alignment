'''
adapted from https://github.com/facebookresearch/dino/blob/main/eval_linear.py
TODO think about handling model_embed_dim again at some point
'''

import torch
from torch import nn

from .dino_urls import dinov1_linear_urls

def build_dino_classifier(model_name, config):
    
    n_last_blocks = config['n_last_blocks']
    avgpool_patchtokens = config['avgpool']
    model_embed_dim = config['model_embed_dim']
    embed_dim = model_embed_dim * (n_last_blocks + int(avgpool_patchtokens))
    linear_classifier = LinearClassifier(embed_dim, num_labels=1000)

    if config['version'] == 'v1':
        url = dinov1_linear_urls[model_name]
    else: 
        raise Exception("only DINO V1 implemented yet")
    
    state_dict = torch.hub.load_state_dict_from_url(url)["state_dict"]    
    linear_classifier.load_state_dict(state_dict, strict=True)

    linear_classifier.eval()
    return linear_classifier


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
