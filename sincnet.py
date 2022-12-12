import tensorflow as tf
from sincnet.layers.tf import SincConv

# Define a model that takes an input tensor of shape (batch_size, 1, num_samples)
# and applies a SincNet layer followed by a ReLU activation and a max pooling layer
class SincNetModel(tf.keras.Model):
    def __init__(self, num_filters, kernel_size, pool_size):
        super(SincNetModel, self).__init__()
        
        self.sinc_layer = SincConv(1, num_filters, kernel_size, sample_rate=16000)
        self.relu = tf.keras.layers.ReLU()
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size)
        
    def call(self, x):
        x = self.sinc_layer(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x
        
# Create an instance of the model
model = SincNetModel(num_filters=64, kernel_size=251, pool_size=3)

# Apply the model to an input tensor
x = tf.random.normal(shape=(16, 1, 10000))
output = model(x)

# Print the shape of the output tensor
print(output.shape)
