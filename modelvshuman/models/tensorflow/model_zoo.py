from ..registry import register_model

from ..wrappers.tensorflow import TensorflowModel
from .build_model import build_model_from_hub

'''
Imports for harmonized architectures
the following must be added to requirements:
efficientnet
'''

import tesnorflow as tf
from .harmonized_model_urls import harmonized_urls
import efficientnet.keras


@register_model("tensorflow")
def efficientnet_b0(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)


@register_model("tensorflow")
def resnet50(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)


@register_model("tensorflow")
def mobilenet_v1(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)


@register_model("tensorflow")
def inception_v1(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def inception_v1(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)

'''
HARMONIZED MODELS
'''
'''
@register_model("tensorflow")
def convnext_harmonized(model_name, *args):
    model = build_model_from_harmonized_model_collection(model_name)
    return TensorflowModel(model, model_name, *args)
'''
@register_model("tensorflow")
def efficient_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("efficientnetB0_harmonized",
                                           harmonized_urls('efficient_harmonized'),
                                           cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)
    return TensorflowModel(model, model_name, *args)
'''
@register_model("tensorflow")
def vit_harmonized(model_name, *args):
    model = build_model_from_harmonized_model_collection(model_name)
    return TensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def levit_harmonized(model_name, *args):
    model = build_model_from_harmonized_model_collection(model_name)
    return TensorflowModel(model, model_name, *args)
'''
@register_model("tensorflow")
def resnet_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("resnet50v2_harmonized", harmonized_urls('resnet_harmonized'),
                                            cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)
    return TensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def vgg_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("vgg16_harmonized", harmonized_urls('vgg_harmonized'),
                                            cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)
    return TensorflowModel(model, model_name, *args)
'''
@register_model("tensorflow")
def vit_harmonized(model_name, *args):
    model = build_model_from_harmonized_model_collection(model_name)
    return TensorflowModel(model, model_name, *args)
'''



