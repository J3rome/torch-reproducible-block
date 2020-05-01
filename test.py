import random
import torch
import numpy as np

from reproducible_block import Reproductible_Block

# TODO : Test random operation Cuda tensors
@Reproductible_Block(42)
def test1(param1, param2):
    print(f"test 1 Random value : {random.random()}")
    print(f"test 1 Torch Random value : {torch.rand((1))[0]}")
    print(f"test 1 Numpy random value : {np.random.random()}")
    print(f"Param 1 : {param1}")
    print(f"Param 2 : {param2}")

@Reproductible_Block(42)
def test2():
    print(f"test 2 Random value : {random.random()}")
    print(f"test 2 Torch Random value : {torch.rand((1))[0]}")
    print(f"test 2 Numpy random value : {np.random.random()}")


def test3():
    with Reproductible_Block(42):
        print(f"test 3 Random value : {random.random()}")
        print(f"test 3 Torch Random value : {torch.rand((1))[0]}")
        print(f"test 3 Numpy random value : {np.random.random()}")


Reproductible_Block.set_seed(42)
test1("42", "Is the answer")
test2()
test3()
