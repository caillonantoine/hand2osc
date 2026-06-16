from typing import Any, Callable

import jax.numpy as jnp
import tensorflow as tf
from jax.experimental import jax2tf


def convert_jax_to_litert(jax_fn: Callable[[Any], Any],
                          d_in: int,
                          dtype: Any = jnp.float32) -> bytes:
    """Converts a 1xD JAX function to LiteRT bytes.
    
    Args:
        jax_fn: A JAX function taking a (1, D) array and returning a (1, D') array.
        d_in: The dimension size 'D' of the input feature space.
        dtype: The JAX data type of the input array.
        
    Returns:
        The serialized LiteRT (.tflite) model as bytes.
    """
    # 1. Convert the JAX function into a TensorFlow-compatible graph function
    tf_predict = jax2tf.convert(jax_fn, with_gradient=False)

    # 2. Map the JAX data type to its corresponding TensorFlow type
    tf_dtype = tf.as_dtype(jnp.dtype(dtype).name)

    # 3. Create a trackable TF module wrapper to strictly bind the input signature
    class LiteRTModelWrapper(tf.Module):

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(1, d_in), dtype=tf_dtype, name="input_tensor")
        ])
        def __call__(self, x):
            return tf_predict(x)

    # 4. Extract the concrete function graph
    tf_module = LiteRTModelWrapper()
    concrete_func = tf_module.__call__.get_concrete_function()

    # 5. Build and run the LiteRT Converter pipeline
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func], trackable_obj=tf_module)

    # Optional performance optimization flags for LiteRT edge execution
    converter.optimizations = {tf.lite.Optimize.DEFAULT}

    return converter.convert()