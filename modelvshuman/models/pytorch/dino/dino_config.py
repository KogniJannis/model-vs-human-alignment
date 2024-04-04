
'''
#TODO
maybe just move this to the model registration?
or if I leave this here maybe integrate urls?
'''


'''
last_blocks and avgpoo_patchtokens are CLI parameters in the original implementation:
n_last_blocks: 
    default=4
    type=int
    "Concatenate [CLS] tokens for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.
avgpool_patchtokens:
    default=False
    type=utils.bool_flag
    "Whether ot not to concatenate the global average pooled features to the [CLS] token. We typically set this to False for ViT-Small and to True with ViT-Base."
'''
    
def load_dino_config(model, model_name):
    if model_name == 'dino_vits16_linear':
        config = {
            'version':'v1',
            'arch':'vit',
            'avgpool':False,
            'n_last_blocks':4,
            'model_embed_dim':model.embed_dim,
        }
    elif model_name == 'dino_vits8_linear':
        config = {
            'version':'v1',
            'arch':'vit',
            'avgpool':False,
            'n_last_blocks':4,
            'model_embed_dim':model.embed_dim,
        }
    elif model_name == 'dino_vitb16_linear':
        config = {
            'version':'v1',
            'arch':'vit',
            'avgpool':True,
            'n_last_blocks':1,
            'model_embed_dim':model.embed_dim,
        }
    elif model_name == 'dino_vitb8_linear':
        config = {
            'version':'v1',
            'arch':'vit',
            'avgpool':True,
            'n_last_blocks':1,
            'model_embed_dim':model.embed_dim,
        }
    elif model_name == 'dino_resnet50_linear':
        '''
        command example in the repo: python eval_linear.py --evaluate --arch resnet50 --data_path /path/to/imagenet/train
        -> seems to indicate that the default values for avgpool and n_last_blocks are to be used
        '''

        config = {
            'version':'v1',
            'arch':'resnet',
            'avgpool':False,
            'n_last_blocks':4,
            'model_embed_dim':model.fc.weight.shape[1],
        }
    else:
        raise Exception(f"The DINO mode {model_name} has no predefined classifier settings yet. For now this throws an exception")
    
    return config