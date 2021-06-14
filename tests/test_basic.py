import pytest
import random
import torch
import numpy as np
from reproducible_block import Reproducible_Block


def setup_module(module):
    """
    Set random seed and set reference random state before running the tests
    """
    seed_value = 42
    print(f"[INFO] Setting seed to {seed_value}")
    Reproducible_Block.set_seed(seed_value)


@pytest.fixture
def teardown():
    """
    Revert modification to the random state by the test
    """
    yield
    Reproducible_Block.reset_to_reference_state()


def generate_random_numbers():
    """
    Do random number generation (Python, PyTorch, Numpy)
    """
    return random.random(), torch.rand((1))[0], np.random.random()


def test_function_decorator_with_same_block_seed_should_give_same_result(teardown):
    @Reproducible_Block(block_seed=42)
    def with_block_seed_42_first():
        return generate_random_numbers()

    @Reproducible_Block(block_seed=42)
    def with_block_seed_42_second():
        return generate_random_numbers()

    first_random_values_with_modifier_42 = with_block_seed_42_first()
    second_random_values_with_modifier_42 = with_block_seed_42_second()

    assert first_random_values_with_modifier_42 == second_random_values_with_modifier_42, \
        "Function decorator with same block seed should generate same value"


def test_function_decorator_with_different_block_seed_should_give_different_result(teardown):
    @Reproducible_Block(block_seed=42)
    def with_block_seed_42():
        return generate_random_numbers()

    @Reproducible_Block(block_seed=64)
    def with_block_seed_64():
        return generate_random_numbers()

    random_values_with_modifier_42 = with_block_seed_42()
    random_values_with_modifier_64 = with_block_seed_64()

    assert random_values_with_modifier_42 != random_values_with_modifier_64, \
        "Function decorator with different block seed should not generate same value"


def test_with_clause_with_same_block_seed_should_give_same_result(teardown):
    with Reproducible_Block(block_seed=42):
        first_random_values_with_modifier_42 = generate_random_numbers()

    with Reproducible_Block(block_seed=42):
        second_random_values_with_modifier_42 = generate_random_numbers()

    assert first_random_values_with_modifier_42 == second_random_values_with_modifier_42, \
        "With clause with same block seed should generate same value"


def test_with_clause_with_different_block_seed_should_give_different_result(teardown):
    with Reproducible_Block(block_seed=42):
        random_values_with_modifier_42 = generate_random_numbers()

    with Reproducible_Block(block_seed=64):
        random_values_with_modifier_64 = generate_random_numbers()

    assert random_values_with_modifier_42 != random_values_with_modifier_64, \
        "With clause with different block seed should not generate same value"


def test_with_clause_and_function_decorator_have_same_behaviour(teardown):
    @Reproducible_Block(block_seed=42)
    def with_block_seed_42():
        return generate_random_numbers()

    decorator_random_values_with_modifier_42 = with_block_seed_42()

    with Reproducible_Block(block_seed=42):
        with_clause_random_values_with_modifier_42 = generate_random_numbers()

    assert decorator_random_values_with_modifier_42 == with_clause_random_values_with_modifier_42, \
        "Decorator function should behave the same as with clause"


def test_block_seed_0_does_reset_to_initial(teardown):
    first_random_values = generate_random_numbers()
    second_random_values = generate_random_numbers()

    with Reproducible_Block(block_seed=0):
        first_random_values_in_block = generate_random_numbers()
        second_random_values_in_block = generate_random_numbers()

    assert first_random_values == first_random_values_in_block and \
        second_random_values ==  second_random_values_in_block, \
        "Using a block seed of 0 should reset to the initial random state"


def test_reset_after_block_doesnt_affect_random_state(teardown):

    first_random_values = generate_random_numbers()
    second_random_values = generate_random_numbers()
    third_random_values = generate_random_numbers()


    # Let's reset the state to the initial value to compare what would happen when we insert a block
    Reproducible_Block.reset_to_reference_state()

    first_random_values_before_block = generate_random_numbers()
    second_random_values_before_block = generate_random_numbers()

    # This block should not affect the random state because random state is reseted 
    # to the value it had when entering the block on exit
    with Reproducible_Block(block_seed=42, reset_state_after=True):
        dummy_random_values = generate_random_numbers()
        dummy_random_values = generate_random_numbers()

    third_random_values_after_block = generate_random_numbers()

    assert third_random_values == third_random_values_after_block, \
        "Block with reset_state_after should not affect the random generator"
