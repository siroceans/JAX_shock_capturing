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
steps = 1000
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

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

testloader = torch.utils.data.DataLoader(
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
        key1, key2, key3, key4 = jax.random.split(key, 4)

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
            jax.nn.relu, # relu activation
            eqx.nn.Linear(64, 10, key = key4), # final fc output matches dimension of classes
            jax.nn.log_softmax, # turns raw outputs to log probabilities
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]: # jaxtyping annotations
        for layer in self.layers: 
            x = layer(x) # forward prop: walk through the layers and override x with the output
        return x

key, subkey = jax.random.split(key, 2) # split the key into two keys, use the subkey rn
model = CNN(subkey)

#print(model)

def cross_entropy(y:Int[Array, "batch"], 
                  pred_y:Float[Array, "batch 10"]) -> Float[Array, ""]:
    # y are the true targets (integers from 1 to 9)
    # pred_y are the log-softmax'd predictions, size (batch, 10)
    # this means: 
    #       There are 10 probabilities per image (one prob per digit). We only care about the 
    #       probability assigned to the CORRECT class (this should be the one class that is correct)
    #       So: for image i we want to find pred_y[i, y[i]], doing it for all 64 images at once
    #       this is what jnp.take_along_axis does!!!!!!!

    y = jnp.expand_dims(y, 1) # resizing from (batch,) to (batch, 1) to match size expectations 
    pred_y = jnp.take_along_axis(pred_y, y, axis = 1)
    return -jnp.mean(pred_y)

def loss(model: CNN, 
        x: Float[Array, "batch 1 28 28"], 
        y: Int[Array, "batch"]) -> Float [Array, ""]:
    # input size is (batch, 1, 28, 28)
    # CNN model operates on input size (1, 28, 28) [i.e. on a single image]
    # so we need to vectorize the model to work on the leading (batch) axis!!!
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)

## Example loss
#loss_value = loss(model, dummy_x, dummy_y)
#print(loss_value.shape)  # scalar loss

## Example inference
#output = jax.vmap(model)(dummy_x)
#print(output.shape)  # batch of predictions

#########################################################
#  evaluation functions
#########################################################

# first we will jit our loss function from earlier !!!
#       Our model (really a pytree) contains leaves that are not arrays (like relu and other
#       activations). When jax creates the tracers for BOTH grad'ing and jit'ing, it makes tracers
#       for ALL leaves in the pytree, but some of these are not arrays so jax fails...
#       HENCE, we need to filter the functions first and then grad or jit...

## TLDR: filtering decides what counts as a traced "input" to the compiled function, vs what is 
#  actual fixed structure of the function. 

loss = eqx.filter_jit(loss)

@eqx.filter_jit
def compute_accuracy(model:CNN, 
                     x:Float[Array, "batch 1 28 28"], 
                     y:Int[Array, "batch"]) -> Float[Array, ""]:
    """
    function takes as input the current model and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1) # grabbing index of highest probability (result from model)
    return jnp.mean(y == pred_y)

def evaluate(model: CNN, 
             testloader:torch.utils.data.DataLoader):
    """function that evaluates the model on the test dataset, computing both the average loss and 
    the average accuracy. 
    """

    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # all the jax operations happen inside 'loss' and 'compute_accuracy' functions, which are 
        # jit'ed, so this is fast!
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)

    avg_loss = avg_loss / len(testloader)
    avg_acc = avg_acc / len(testloader)
    return avg_loss, avg_acc

## example evaluation of the network
#loss, acc = evaluate(model, testloader)
#print(loss)
#print(acc)

#########################################################
# Training the network 
#########################################################

optimizer = optax.adamw(learning_rate)

def train(model:CNN, 
          trainloader: torch.utils.data.DataLoader, 
          testloader: torch.utils.data.DataLoader, 
          optim: optax.GradientTransformation, 
          steps: int, 
          print_every: int,) -> CNN: 
    # again, it only makes sense to train the parameters (i.e. the arrays) in our model, 
    # so we filter out the rest

    #------------------------------------------------------------------------------------
    # How does the optax api work? It boils down to three functions
    #       - optim.init(params) -> build the initial opt_state shaped to trainable parameters. imp-
    #       ortant to remember about jax functional model, so optimization states have to be stored
    #       and passed to each iteration step as a PyTree
    #       - optim.update(grads, opt_state, params) -> compute parameter updates and new opt_state
    #       - eqx.apply_updates(model, updates) -> eqx helper that does new_param = old + update for 
    #       every single array leaf! hands back a whole new model
    #------------------------------------------------------------------------------------

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # wrap it all into a single jit'ed region!!
    @eqx.filter_jit
    def make_step(model:CNN, 
                  opt_state: PyTree, 
                  x: Float[Array, "batch 1 28 28"], 
                  y: Int[Array, "batch"]):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # loop over training dataset as many times as needed...
    def infinite_trainloader(): 
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()): 
        # convert pytorch tensors to np arrays
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, test_loss={test_loss.item()}, "
                f"test_accuracy={test_accuracy.item()}"
            ) 
    return model

model = train(model, trainloader, testloader, optimizer, steps, print_every)