import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import optax # for training of the network
import torch # loading mnist dataset
import torchvision # loading mnist dataset

#########################################################
# Hyperparameters
#########################################################

batch_size = 64
learning_rate = 3e-4
steps = 300
print_every = 30
seed = 5678

key = jax.random.PRNGKey(seed)

#########################################################
# Loading dataset using pytorch: 
#   will not be using pytorch for this myself, so no need
#   to spend lots of time really learning this. 
#########################################################

normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = torchvision.datasets.MNIST(
    "MNIST", 
    train = True, 
    download = True, 
    transform = normalise_data,
) 

test_dataset = torchvision.datasets.MNIST(
    "MNIST", 
    train = False, 
    download = True, 
    transform = normalise_data,
)

trainloader = torch.utils.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

testloader = torch.utils.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

# checking data
dummy_x, dummy_y = next(iter(trainloader)) # dummy_x: batch of images; dummy_y: batch of labels
dummy_x = dummy_x.numpy() # transform torch tensors to np arrays
dummy_y = dummy_y.numpy()
# print(dummy_x.shape) # ->  (64, 1, 28, 28) (batch, channels, resolution, resolution)
# print(dummy_y.shape) # ->  (64,) (batch)


#########################################################
# Creating the model
#########################################################

class CNN(eqx.Module): # CNN is a subclass of eqx.module!! (remember inheritance?)
    layers: list # telling python the layers attribute will be of type list. 

    def __init__(self, key): # constructor method for the class, self is passed automatically and
        # key has to be provided
        key1, key2, key3, key4 = jax.random.split(key, 4)and

        # CNN setup: conv layer, followed by flattening, with a small MLP on top!
        self.layers = [ # before, we defined layers to be of type list, so now we actually 
            # create this list!! 
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1), # (input channel, output channels, ,)
            eqx.nn.MaxPool2d(kernel_size=2), 
            jax.nn.relu, # relu activation
            jnp.ravel, # ravel flattens 2d array into 1d
            eqx.nn.Linear(1728, 512, key=key2), # fc layer with 1728 input to 512 output, weights randomly initialized with key 2
            jax.nn.sigmoid, # sigmoid activation
            eqx.nn.Linear(512, 64, key = key3), # fc layer
            eqx.nn.relu, # relu activation
            eqx.nn.Linear(64, 10, key = key4), # final fc output matches dimension of classes
            jax.nn.log_softmax, # turns raw outputs to log probabilities
        ]

        def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]: 
