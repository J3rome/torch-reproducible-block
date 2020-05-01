import random

import torch
import numpy as np


# TODO : Add some comments
# TODO : PYTHONHASHSEED for reproductible dictionaries order -- https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
class Reproductible_Block:
    """
    Python, Numpy and PyTorch random states manager statement
    """
    # Class attribute
    reference_state = None

    def __init__(self, block_seed=0, reset_state_after=False):
        if callable(block_seed):
            # FIXME : Not sur RuntimeError is the best exception
            raise RuntimeError("Block seed must be passed to the decorator Ex: @Reproducible_Block(block_seed=42)")

        self.block_seed = block_seed
        self.reset_state_after = reset_state_after
        self.initial_state = None

    # With clause handling
    def __enter__(self):
        if self.reset_state_after:
            self.initial_state = get_random_state()

        # TODO : Add warning when reference_state is not set ?
        if Reproductible_Block.reference_state:
            set_random_state(Reproductible_Block.reference_state)

        # Modify the random state by performing a serie of random operations.
        # This is done to create unique random state paths using the same initial state for different code blocks
        # If we were to set the same state before every operation, the same operation at different stage of the model
        # would always have the same result (Ex : Weight initialisation)
        modify_random_state(self.block_seed)

    # With clause handling
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.reset_state_after:
            set_random_state(self.initial_state)

    # Decorator handling
    def __call__(self, fct, *args):
        def wrapped_fct(*args):
            with self:
                fct(*args)
        return wrapped_fct

    @classmethod
    def set_seed(cls, seed):
        set_random_seed(seed)

        cls.set_reference_state()

    @classmethod
    def set_reference_state(cls):
        cls.reference_state = get_random_state()

    @staticmethod
    def get_random_state():
        # Helper method so we can import only this class
        return get_random_state()

    @staticmethod
    def set_random_state(state):
        # Helper method so we can import only this class
        set_random_state(state)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # FIXME : Use manual_seed_all ? (For multiple gpu usecase)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_random_state():
    states = {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state()
    }

    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state()

    return states


def set_random_state(states):
    random.setstate(states['py'])
    np.random.set_state(states['np'])
    torch.random.set_rng_state(states['torch'])

    if torch.cuda.is_available() and 'torch_cuda' in states:
        torch.cuda.set_rng_state(states['torch_cuda'])


def modify_random_state(modify_seed):
    # Modify the random state by performing a serie of random operations.
    for i in range(modify_seed):
        random.randint(1, 10)
        np.random.randint(1, 10)

        # TODO : Check if both torch & torch.cuda random state are impacted by this. Might need to do operation on cuda tensor
        torch.randint(1, 10, (1,))