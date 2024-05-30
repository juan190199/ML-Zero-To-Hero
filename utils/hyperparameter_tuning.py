import random
from itertools import product
import numpy as np

ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


# def grid_search(train_loader,
#                 val_loader,
#                 model_class,
#                 grid_search_spaces={
#                     'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]
#                 }, epochs=20, patience=5):
#     """
#     Grid search based on nested loops for hyperparameter tuning
#     Args:
#         train_loader:
#         val_loader:
#         model_class:
#         grid_search_spaces:
#         epochs:
#         patience:
#
#     Returns:
#
#     """
