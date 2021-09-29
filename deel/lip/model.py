# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains equivalents for Model and Sequential. These classes add support
for condensation and vanilla exportation.
"""
import math
from typing import Dict
from warnings import warn

import numpy as np
from tensorflow import Tensor
from tensorflow.keras import Sequential as KerasSequential, Model as KerasModel
from tensorflow.keras.layers import Input, InputLayer
from tensorflow.keras.utils import register_keras_serializable

from .layers import LipschitzLayer, Condensable


@register_keras_serializable("deel-lip", "Sequential")
class Sequential(KerasSequential, LipschitzLayer, Condensable):
    def __init__(
        self,
        layers=None,
        name=None,
        k_coef_lip=1.0,
    ):
        """
        Equivalent of keras.Sequential but allow to set k-lip factor globally. Also
        support condensation and vanilla exportation.
        For now constant repartition is implemented (each layer
        get n_sqrt(k_lip_factor), where n is the number of layers)
        But in the future other repartition function may be implemented.

        Args:
            layers: list of layers to add to the model.
            name: name of the model, can be None
            k_coef_lip: the Lipschitz coefficient to ensure globally on the model.
        """
        super(Sequential, self).__init__(layers, name)
        self.set_klip_factor(k_coef_lip)

    def build(self, input_shape=None):
        self._init_lip_coef(input_shape)
        return super(Sequential, self).build(input_shape)

    def set_klip_factor(self, klip_factor):
        super(Sequential, self).set_klip_factor(klip_factor)
        nb_layers = np.sum([isinstance(layer, LipschitzLayer) for layer in self.layers])
        for layer in self.layers:
            if isinstance(layer, LipschitzLayer):
                layer.set_klip_factor(math.pow(klip_factor, 1 / nb_layers))
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitz layer: {}".format(  # noqa: E501
                        layer.name
                    )
                )

    def _compute_lip_coef(self, input_shape=None):
        for layer in self.layers:
            if isinstance(layer, LipschitzLayer):
                layer._compute_lip_coef(input_shape)
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitz layer: {}".format(  # noqa: E501
                        layer.name
                    )
                )

    def _init_lip_coef(self, input_shape):
        for layer in self.layers:
            if isinstance(layer, LipschitzLayer):
                layer._init_lip_coef(input_shape)
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitz layer: {}".format(  # noqa: E501
                        layer.name
                    )
                )

    def _get_coef(self):
        global_coef = 1.0
        for layer in self.layers:
            if isinstance(layer, LipschitzLayer) and (global_coef is not None):
                global_coef *= layer._get_coef()
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitz layer: {}".format(  # noqa: E501
                        layer.name
                    )
                )
                global_coef = None
        return global_coef

    def condense(self):
        for layer in self.layers:
            if isinstance(layer, Condensable):
                layer.condense()

    def vanilla_export(self):
        layers = list()
        for layer in self.layers:
            if isinstance(layer, Condensable):
                layers.append(layer.vanilla_export())
            else:
                lay_cp = layer.__class__.from_config(layer.get_config())
                lay_cp.build(layer.input.shape[1:])
                lay_cp.set_weights(layer.get_weights())
                layers.append(lay_cp)
        model = KerasSequential(layers, self.name)
        return model

    def get_config(self):
        config = {"k_coef_lip": self.k_coef_lip}
        base_config = super(Sequential, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "Model")
class Model(KerasModel):
    """
    Equivalent of keras.Model but support condensation and vanilla exportation.

    Warning:
         As lipschitz constant are multiplicative along layer, the Model class
         cannot set a global Lipschitz constant (problem with branching inside a
         model).
    """

    def condense(self):
        for layer in self.layers:
            if isinstance(layer, Condensable):
                layer.condense()

    def vanilla_export(self) -> KerasModel:
        """
        Export this model to a "Vanilla" model, i.e. a model without Condensable
        layers.

        Returns:
            A Keras model, identical to this model, but where condensable layers have
            been replaced with their vanilla equivalent (e.g. SpectralConv2D with
             Conv2D).
        """
        # Dictionary that will map tensor names (from the current model) to tensors
        # in the exported model.# We initialize the dictionary for inputs:
        tensors: Dict[str, Tensor] = {}

        # Initialize the dictionary with inputs:
        tensors.update({inp.name: Input(shape=inp.shape[1:]) for inp in self.inputs})

        for lay in self.layers:

            # Skip input layers:
            if isinstance(lay, InputLayer):
                continue

            # Condense+Export the layer if it is a non-vanilla layer, otherwise
            # just copy the layer:
            if isinstance(lay, Condensable):
                lay_cp = lay.vanilla_export()
            else:
                # Duplicate layer (weights are not duplicated):
                lay_cp = lay.__class__.from_config(lay.get_config())
                lay_cp.build(lay.input_shape)
                lay_cp.set_weights(lay.get_weights().copy())

            # For each input nodes, we are going to create corresponding operations
            # in the exported models:
            for inode in range(len(lay.inbound_nodes)):
                inputs = lay.get_input_at(inode)
                outputs = lay.get_output_at(inode)

                # Fetch the
                if isinstance(inputs, list):
                    inputs = [tensors[input.name] for input in inputs]
                else:
                    inputs = tensors[inputs.name]

                # Retrieve outputs layers (for the exported layer):
                moutputs = lay_cp(inputs)

                # Add the output tensors to the dictionary, using the names from the
                # original model:
                if isinstance(outputs, list):
                    for outi, mouti in zip(outputs, moutputs):
                        tensors[outi.name] = mouti
                else:
                    tensors[outputs.name] = moutputs

        return KerasModel(
            [tensors[inp.name] for inp in self.inputs],
            [tensors[out.name] for out in self.outputs],
        )


vanillaModel = Model.vanilla_export
