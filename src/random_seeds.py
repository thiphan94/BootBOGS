import numpy as np
import tensorflow as tf
import random 


def reset_seeds(number=42,reset_graph_with_backend=None):
    """
    Reset random seeds number
    """
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional

    np.random.seed(number)
    random.seed(number)
    tf.compat.v1.set_random_seed(number)
    # print("RANDOM SEEDS RESET")  # optional
    