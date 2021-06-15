# Torch-reproducible-block

Random number generation is hard to control when experimenting with neural networks. Setting a random seed only get us so far. Each random operation affects the random number generator state. Changes to the model hyper-parameters or architecture can affect how each layer is initialised, the regularisation techniques, how the data is presented to the network during training and more.



This package aims to reduce **variability** by limiting **side effects** caused by **random number generation**.

The main goal is to **limit the changes** to the rest of the network when **experimenting** with different hyper-parameters.

---

## What is the problem ?

The weight initialisation of a layer constitute a random operation. The initialisation order therefore have an impact on subsequent layers.

![](img/problem.png)



In this small toy model, the **initial weights** of the fully connected layers will be **different** if we have a different number of convolutive layers.

The initialisation of a **pre-trained feature extractor** might also comprises random operation which will **affect the rest of the network**. 

A different random state will also affect the **dataloading** process since it also rely on random operations to select random examples when creating batches.



## Solution

We isolate different parts of the network by wrapping them inside a `Reproducible_Block` 

![](img/solution.png)



## How does it work ?

`Reproducible_Block.set_seed(SEED_VALUE)` must be called **before** any random operation. 

This will set the `python`, `numpy` and `torch` seeds and will save a copy of the `initial random number generator state`.



When entering a `Reproducible_Block` the random number generator state is reset to the `initial state`. 

The state is then mutated according to the `Block Seed` value ensuring that each block have a different state. 

To mutate the state, we simply run `X` random operations where `X` is `Block Seed`.

Feel free to take at look at the code, it's only about 100 lines.

---

## Installation

The package can be installed via pip :

```bash
pip install torch-reproducible-block
```

The following packages are required :

```
torch
numpy
```



## Usage

```python
from reproducible_block import Reproducible_Block

class My_model(nn.Module):
    def __init__(self):
        with Reproducible_Block(block_seed=64):
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[2,2])
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[2,2])

        with Reproducible_Block(block_seed=44):
            self.fc = nn.Linear(64, 128)
            self.out = nn.Linear(128, 10)
            
    def forward(self, batch):
        ...
    
if __name__ == "__main__":
    # Will set seed and save the initial random state
	Reproducible_Block.set_seed(42)
    
    model = My_model()
    # Data loading and other configurations which might do random operations....
    
    # Ensure that we always have the same random state when starting training
    with Reproducible_Block(block_seed=128):
        train_model(model, data, ...)
```



Reproducible block can also be used as a function decorator  :

```python
@Reproducible_Block(block_seed=128)
def train_model(model, dataloader, ...):
    ...
```



## Remarks

- Using a different `initial seed` (Via `Reproducible_Block.set_seed()`), will result in a different random state for each `Reproducible_Block`.
- The `block seed` is "part of the code". You should not attempt to tweak it the way we do with "normal seed". Changing the `initial seed` is what you want to do in order to create different initial conditions for your training.
- Using the same `block seed` for different `Reproducible_Block` will result in the same random state. 
  Make sure that you are using a different `block seed` for each block.
- Was tested on `Ubuntu 18.04` with `python 3.6`. Should not have any problems running on other platforms. Fill up an issue if you have any problems.

## Other sources of randomness

This package won't make your research 100% reproducible. It simply aim to isolate part of your program to side effects.

- Setting the `PYTHONHASHSEED` environment variable is always a good idea.
- The way python read files from directories (ex : `os.listdir()`)  can vary when ran on different OS (Linux vs Windows)
- Different version of `python`, `Cuda`, `Cudnn` and libraries can affect reproducibility

## Contributing

The code is pretty simple, have a look at it and if you have improvements ideas feel free to submit a pull request !

