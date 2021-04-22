from flax import linen as nn

class SimpleCNN(nn.Module):
    """Very simple convolutional neural network."""
    num_classes: int
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=32, kernel_size=(3,3), strides=(1,1), dtype=x.dtype)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1)) # flatten all except batch dim
        x = nn.Dense(128, dtype=x.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes, dtype=x.dtype)(x)
        return nn.log_softmax(x)
