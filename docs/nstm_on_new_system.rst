Build your own nstm reconstruction
==================================

This tutorial will guide you through the process of building your own space-time reconstruction using nstm method.

Write forward model
-------------------

NSTM relies on a **deterministic and differentiable forward model** to get meaningful gradients to update the model's weights during reconstruction.

Here we will use a simple linear model to demonstrate the process. The forward model is defined as:

    .. math::
        y = x \cdot w

where :math:`x` is the object to recover and :math:`w` is a pre-defined weight matrix. The forward model is implemented as follows:

    .. code-block:: python

        import jax.numpy as jnp
        import calcil as cc

        class NewImager(cc.forward.Model):
            w: jnp.ndarray  # pre-defined weight matrix

            def setup():
                # nothing to do here
                pass

            def __call__(self, x):
                """Forward model"""
                return x @ self.w

The forward model is a class that inherits from ``cc.forward.Model``. The `setup` method is used to initialize the model's parameters. The ``__call__`` method is the forward model itself. The forward model takes an input ``x`` and returns the output ``y``.

Combine with space-time model
-----------------------------

Once the forward model is defined, we can combine it with the nstm method to render the measurement at different timepoints.

        .. code-block:: python

            from nstm import spacetime

            class NewImagerWithNSTM(cc.forward.Model):
                w: jnp.ndarray  # pre-defined weight matrix
                spacetime_param: spacetime.SpaceTimeParameters  # nstm parameters

                def setup():
                    self.forward = NewImager(w=self.w)  # Initialize the forward model

                    # Initialize the nstm reconstruction
                    self.spacetime = spacetime.SpaceTimeMLP(optical_param=self.w.shape,  # specify the shape and dim of the nstm reconstruction
                                                            spacetime_param=self.spacetime_param,  # nstm parameters
                                                            num_output_channels=1)  # assume the output is a single channel

                def __call__(self, input_dict):
                    t = input_dict['t']  # given timepoint
                    obj = self.spacetime(t, coord_offset=jnp.zeros((1, 2)))
                    img = self.forward(obj)
                    return img

The ``NewImagerWithNSTM`` class combines the forward model with the nstm method. The ``setup`` method initializes the forward model and the nstm reconstruction.
The ``__call__`` method takes an input timepoint and feeds it into the nstm to get the reconstructed object at that timepoint.
The forward model then takes the reconstructed object and returns the rendered measurement.

.. note::
    Isn't that simple? Now you can build your own nstm reconstruction with your favorite imaging system!

All left is to define a loss function and train the network weights of nstm using your data.
We found that L2 loss between the rendered measurement and the actual measurement works well.

If you're bored with copying and pasting template code to do gradient descent-based image reconstruction, `calcil`_ package can do some brainless work for you 🧠.

.. _calcil: https://github.com/rmcao/CalCIL
