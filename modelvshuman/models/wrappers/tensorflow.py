import numpy as np
import tensorflow as tf
from skimage.transform import resize

tf.compat.v1.enable_eager_execution()
from .base import AbstractModel
from ...helper.human_categories import compute_imagenet_indices_for_category


def get_device(device=None):
    import tensorflow as tf

    if device is None:
        device = tf.device("/GPU:0" if tf.test.is_gpu_available() else "/CPU:0")
    if isinstance(device, str):
        device = tf.device(device)
    return device


class TensorflowModel(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args

    def softmax(self, logits):
        assert type(logits) is np.ndarray
        return tf.nn.softmax(logits).numpy()

    def forward_batch(self, images):
        device = get_device()
        with device:
            predictions = self.model(images)
            return predictions.numpy()

'''
Wrapper for Efficientnets b1-b7 which need image resizing
(I'm scared of putting resize in the standard wrapper; this should ensure max. backward compatibility/reproducibility
'''

class TensorflowEfficientnetModel(TensorflowModel):

    def __init__(self, model, model_name, effnet_resolution, *args):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.effnet_resolution = effnet_resolution

    def softmax(self, logits):
        assert type(logits) is np.ndarray
        return tf.nn.softmax(logits).numpy()

    def forward_batch(self, images):
        device = get_device()
        with device:
            images = tf.image.resize(images, [self.effnet_resolution, self.effnet_resolution])
            predictions = self.model(images)
            return predictions.numpy()



'''
Preprocessing for harmonized models (Fel et al., 2022)
CURRENTLY DISABLED
the Harmonization Repo instructs to apply this before the forward pass but it leads to chance performance
-> preprocessing is probably being already applied elsewhere, but I (Jannis) can't find where

Adapted from: https://github.com/serre-lab/Harmonization/blob/main/harmonization/models/preprocess.py
(preprocess_input was directly adapted with constants being inserted directly; commentary left as is)
see https://serre-lab.github.io/Harmonization/training/ for use case
'''
class HarmonizedTensorflowModel(TensorflowModel):
    def __init(self, model, model_name, *args):
        super(HarmonizedTensorflowModel, self).__init__(model, model_name, *args)


    
    def preprocess_input(images): #disabled 
        """
        Preprocesses images for the harmonized models.
        The images are expected to be in RGB format with values in the range [0, 255].
    
        Parameters
        ----------
        images
            Tensor or numpy array to be preprocessed.
            Expected shape (N, W, H, C).
    
        Returns
        -------
        preprocessed_images
            Images preprocessed for the harmonized models.
        """
        images = images / 255.0
        images = images - np.array([0.485, 0.456, 0.406]) #IMAGENET_MEAN
        images = images /  np.array([0.229, 0.224, 0.225]) #IMAGENET_STD 
        return images
        
    def softmax(self, logits):
        assert type(logits) is np.ndarray
        return tf.nn.softmax(logits).numpy()

    def forward_batch(self, images):
        device = get_device()
        with device:
            #images = self.preprocess_input(images) (see above)
            predictions = self.model(images)
            return predictions.numpy()

'''
Wrapper for Vision Transformers to apply their preprocessing
see:
https://github.com/faustomorales/vit-keras/tree/master/vit_keras
https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/levit
https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/maxvit
'''

class VitModel(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args

    def preprocess_input(X): #probably identical to the normal preprocessing after all?
        return tf.keras.applications.imagenet_utils.preprocess_input(
            X, data_format=None, mode="tf"
        )

    def softmax(self, logits):
        assert type(logits) is np.ndarray
        return tf.nn.softmax(logits).numpy()

    def forward_batch(self, images):
        device = get_device()
        with device:
            images = self.preprocess_input(images)                  
            predictions = self.model(images)
            return predictions.numpy()
        


class EffNetUndoPreprocess(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args

    def preprocess_input(images): #disabled 
        """
        Preprocesses images for the harmonized models.
        The images are expected to be in RGB format with values in the range [0, 255].
    
        Parameters
        ----------
        images
            Tensor or numpy array to be preprocessed.
            Expected shape (N, W, H, C).
    
        Returns
        -------
        preprocessed_images
            Images preprocessed for the harmonized models.
        """
        images = images *  np.array([0.229, 0.224, 0.225]) #IMAGENET_STD 
        images = images +  np.array([0.485, 0.456, 0.406]) #IMAGENET_MEAN
        images = images * 255.0
        

        return images
       
    
    def softmax(self, logits):
        assert type(logits) is np.ndarray
        return tf.nn.softmax(logits).numpy()

    def forward_batch(self, images):
        device = get_device()
        with device:
            images = self.preprocess_input(images)
            predictions = self.model(images)
            return predictions.numpy()

class EffNetDoPreprocess(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args

    def preprocess_input(images): #disabled 
        """
        Preprocesses images for the harmonized models.
        The images are expected to be in RGB format with values in the range [0, 255].
    
        Parameters
        ----------
        images
            Tensor or numpy array to be preprocessed.
            Expected shape (N, W, H, C).
    
        Returns
        -------
        preprocessed_images
            Images preprocessed for the harmonized models.
        """
        images = images / 255.0
        images = images - np.array([0.485, 0.456, 0.406]) #IMAGENET_MEAN
        images = images /  np.array([0.229, 0.224, 0.225]) #IMAGENET_STD 
        return images
       

    def softmax(self, logits):
        assert type(logits) is np.ndarray
        return tf.nn.softmax(logits).numpy()

    def forward_batch(self, images):
        device = get_device()
        with device:
            images = self.preprocess_input(images)
            predictions = self.model(images)
            return predictions.numpy()