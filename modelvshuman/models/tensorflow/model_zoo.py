from ..registry import register_model

from ..wrappers.tensorflow import TensorflowModel
from .build_model import build_model_from_hub

'''
Imports for harmonized architectures
the following must be added to requirements:
efficientnet, vit_keras, keras_cv_attention_models
'''
import tensorflow as tf
from .harmonized_model_url import harmonized_urls
from ..wrappers.tensorflow import HarmonizedTensorflowModel, TensorflowPreprocessingModel
import efficientnet.keras
from vit_keras import vit
from keras_cv_attention_models.levit import LeViT128
from keras_cv_attention_models.maxvit import  MaxViT_Tiny
from keras_cv_attention_models.convnext import ConvNeXtTiny

'''
Imports for the keras.applications zoo
'''
from ..wrappers.tensorflow import TensorflowEfficientnetModel


@register_model("tensorflow")
def tfhub_efficientnet_b0(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def tfhub_resnet50(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)


@register_model("tensorflow")
def tfhub_mobilenet_v1(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)


@register_model("tensorflow")
def tfhub_inception_v1(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)

'''
HARMONIZED MODELS
'''

@register_model("tensorflow")
def convnext_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("tiny_convnext_harmonized", harmonized_urls.get('convnext_harmonized'),
                                            cache_subdir="models")
    model = ConvNeXtTiny(classifier_activation=None, pretrained=None)
    model.load_weights(weights_path)
    return HarmonizedTensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def efficient_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("efficientnetB0_harmonized", harmonized_urls.get('efficient_harmonized'),
                                           cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)
    return HarmonizedTensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def levit_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("levit_small_harmonized", harmonized_urls.get('levit_harmonized'),
                                            cache_subdir="models")

    model = LeViT128(classifier_activation = None, use_distillation = False)
    model.load_weights(weights_path)
    return HarmonizedTensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def maxvit_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("maxvit_tiny_harmonized", harmonized_urls.get('maxvit_harmonized'),
                                            cache_subdir="models")

    model = MaxViT_Tiny(classifier_activation = None, pretrained = None)
    model.load_weights(weights_path)
    return HarmonizedTensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def resnet_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("resnet50v2_harmonized", harmonized_urls.get('resnet_harmonized'),
                                            cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)
    return HarmonizedTensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def vgg_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("vgg16_harmonized", harmonized_urls.get('vgg_harmonized'),
                                            cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)
    return HarmonizedTensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def vit_harmonized(model_name, *args):
    weights_path = tf.keras.utils.get_file("vit-b16_harmonized", harmonized_urls.get('vit_harmonized'),
                                            cache_subdir="models")
    model = vit.vit_b16(
        image_size=224,
        activation='linear',
        pretrained=False,
        include_top=True,
        pretrained_top=False
    )
    model.load_weights(weights_path)
    return HarmonizedTensorflowModel(model, model_name, *args)

'''
Keras.applications zoo
'''
@register_model("tensorflow")
def tf_efficientnet_b0(model_name, *args):
    model = tf.keras.applications.efficientnet.EfficientNetB0()
    effnet_resolution = 224
    return TensorflowEfficientnetModel(model, model_name, effnet_resolution, *args)

@register_model("tensorflow")
def tf_efficientnet_b1(model_name, *args):
    model = tf.keras.applications.efficientnet.EfficientNetB1()
    effnet_resolution = 240
    return TensorflowEfficientnetModel(model, model_name, effnet_resolution, *args)

@register_model("tensorflow")
def tf_efficientnet_b2(model_name, *args):
    model = tf.keras.applications.efficientnet.EfficientNetB2()
    effnet_resolution = 260
    return TensorflowEfficientnetModel(model, model_name, effnet_resolution, *args)

@register_model("tensorflow")
def tf_efficientnet_b3(model_name, *args):
    model = tf.keras.applications.efficientnet.EfficientNetB3()
    effnet_resolution = 300
    return TensorflowEfficientnetModel(model, model_name, effnet_resolution, *args)

@register_model("tensorflow")
def tf_efficientnet_b5(model_name, *args):
    model = tf.keras.applications.efficientnet.EfficientNetB5()
    effnet_resolution = 456
    return TensorflowEfficientnetModel(model, model_name, effnet_resolution, *args)

@register_model("tensorflow")
def tf_efficientnet_b7(model_name, *args):
    model = tf.keras.applications.efficientnet.EfficientNetB7()
    effnet_resolution = 600
    return TensorflowEfficientnetModel(model, model_name, effnet_resolution, *args)

@register_model("tensorflow")
def tf_vgg16(model_name, *args):
    model = tf.keras.applications.vgg16.VGG16()
    preprocessing = tf.keras.applications.vgg16.preprocess_input
    return TensorflowPreprocessingModel(model, model_name, preprocessing, *args)

@register_model("tensorflow")
def tf_convnext_tiny(model_name, *args):
    model = tf.keras.applications.convnext.ConvNeXtTiny()
    preprocessing = tf.keras.applications.
    return TensorflowPreprocessingModel(model, model_name, preprocessing, *args)

@register_model("tensorflow")
def tf_resnet50_v2(model_name, *args):
    model = tf.keras.applications.resnet_v2.ResNet50V2()
    preprocessing = keras.applications.resnet_v2.preprocess_input
    return TensorflowPreprocessingModel(model, model_name, preprocessing, *args)


'''
ViT Baselines for harmonized models
'''
@register_model("tensorflow")
def tf_vit_b16(model_name, *args):
    model = vit.vit_b16(
        image_size=224,
        activation='linear',
        pretrained=True,
        include_top=True,
        pretrained_top=True #TODO check, because changed this
    )
    #TODO preprocessing
    return TensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def tf_levit128(model_name, *args):
    model = LeViT128(classifier_activation = "softmax", use_distillation = False) #TODO right configuration?
    return TensorflowModel(model, model_name, *args)

@register_model("tensorflow")
def tf_maxvit_tiny(model_name, *args):
    model = MaxViT_Tiny(classifier_activation = "softmax") #TODO right configuration?
    return TensorflowModel(model, model_name, *args)
