"""
Transformer based models to perform on image classification tasks

@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LayerNormalization, Add, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from Modules.ImagePositionPatchEmbedding import ImagePosEmbed
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow_addons.activations import gelu
from tensorflow_addons.optimizers import AdamW


class VisionTransformer:
    def __init__(self, input_shape, batch_size, classes, patch_size, stride_num, patch_num, proj_dim, num_heads, stack_num, dropout, model="ViT"):
        """
        Initialize VisionTransformer class which constructs the transformer model for images

        Parameters
        ----------
        input_shape : tuple
          The input shape of the model
        batch_size: integer
          The size of batch to train and update weight at once
        classes: integer
          Number of classes to classify
        patch_size : integer
          The width and height of the patch
        stride_num: integer
          The number of pixels of the sliding window to shift
        patch_num : integer
          Number of total patches
        proj_dim : integer
          The vector size to project each patch to and the base number of neurons each layer trough the network
        num_heads: integer
          The number of attention heads in Multi-Head Self-Attention
        stack_num: integer
          The number of transformer encoder stack(depth of the model)
        dropout: float
          The dropout probability trough the whole network
        model : string
          The parameter to decide which model to use.
          "ViT" is the baseline(for this project) Vision Transformer classification model.

        References
        ----------
        "ViT", https://arxiv.org/abs/2010.11929
        """
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.classes = classes
        self.patch_size = patch_size
        self.stride_num = stride_num
        self.patch_num = patch_num
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.stack_num = stack_num
        self.dropout = dropout
        self.history = None
        if model == "ViT":
            self.model = self.ViT()
        else:
            raise ValueError("Model doesn't exist!")

    def ViT(self):
        """
        Function to construct the baseline(for this project) Vision Transformer classification model.

        Returns
        -------
        model : Keras model class
          The baseline Vision Transformer model itself
        """
        # Create Input layer
        inputs = Input(self.input_shape)
        # Generate Image plus Position Patch Embeddings
        embedding_layer = ImagePosEmbed(self.batch_size, self.patch_size, self.stride_num, self.patch_num, self.proj_dim)
        embeddings = embedding_layer(inputs)
        # Transformer Encoder Stacks
        x = embeddings
        for _ in range(self.stack_num):
            x = self.TransformerEncoder(x)
        # MLP Head Classification
        outputs = self.MLP_head(x[:, 0, :], self.proj_dim)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def TransformerEncoder(self, x):
        """
        Function of the Transformer Encoder in the Vision Transformer model.
        Layers in the order of 1. Layer Normalization, 2. Multi-Head Attention, 3. Layer Normalization, 4. Multi Layer Perceptron.
        With addition skip connections from the input of 1. with the output of 2. as the input of 3.
        and addition skip connections from the input of 3. with the output of 4. as the final output.

        Parameters
        ----------
        x : tensor
          The input of patches embeddings or the output of the previous transformer encoder as input

        Returns
        -------
        context : tensor
          The output of the transformer encoder
        """
        x_norm1 = LayerNormalization(epsilon=1e-6)(x)
        x_mha = MultiHeadAttention(head_size=self.proj_dim, num_heads=self.num_heads, dropout=self.dropout)([x_norm1, x_norm1, x_norm1])
        x_concat1 = Add()([x_mha, x_norm1])
        x_norm2 = LayerNormalization(epsilon=1e-6)(x_concat1)
        x_mlp1 = self.MLP(x_norm2, self.proj_dim)
        x_concat2 = Add()([x_mlp1, x_concat1])

        return x_concat2

    def MLP(self, inputs, units):
        """
        Function of the Multi Layer Perceptron in the Transformer Encoder.
        Layers in the order of dense layer with gelu activation, dropout, dense layer with gelu activation, dropout

        Parameters
        ----------
        inputs : tensor
          The input of a tensor in the transformer encoder
        units : tensor
          The parameter to determine the number of units in the dense layers

        Returns
        -------
        context : tensor
          The output of the multi layer perceptron
        """
        x1 = Dense(units*2, activation=gelu)(inputs)
        x1 = Dropout(self.dropout)(x1)
        x2 = Dense(units, activation=gelu)(x1)
        x2 = Dropout(self.dropout)(x2)

        return x2

    def MLP_head(self, inputs, units):
        """
        Function of the Multi Layer Perceptron Head in the Vision Transformer.
        This is the final layers to output the classes of the image.
        Layers in the order of dense layer with gelu activation, dropout, dense layer with linear activation

        Parameters
        ----------
        inputs : tensor
          The input to the MLP_head would be the corresponding output of the Transformer Encoder from the input of
          the class token(This is to not introduce any bias towards any of the patches).
        units : tensor
          The parameter to determine the number of units in the dense layers

        Returns
        -------
        context : tensor
          The final output of the vision transformer without logits(Need additional sigmoid or softmax to get probability)
        """
        x1 = Dense(units*2, activation=gelu)(inputs)
        x1 = Dropout(self.dropout)(x1)
        x2 = Dense(self.classes)(x1)

        return x2

    def summary(self):
        """
        Print the summary of the current transformer model in VisionTransformer class
        """
        self.model.summary()

    def train(self, X_train, X_val, y_train, y_val, optimizer, lr, loss, metrics, epochs, lr_decay=None, decay_rate=0.985, weight_decay=0.00001, save_model=False, save_path=None, monitor='loss', mode='min'):
        """
        Function to train the current transformer model in VisionTransformer class

        Parameters
        ----------
        X_train : tensor
          The train set of tensors of the original RGB images
        X_val : tensor
          The validation set of tensors of the original RGB images
        y_train : tensor
          The train set class labels of tensors of the original RGB images
        y_val : tensor
          The validation set class labels of tensors of the original RGB images
        optimizer : string
          The parameter to decide which optimizer to use.
          "adam" is using the Adam optimizer.
          "adamW" is using the Adam Decouple Weight Decay optimizer.
        lr : float
          The parameter of the initial learning rate
        loss : function
          The loss function used for training
        metrics : list
          A list of metric functions to evaluate train and validation when training
        epochs : integer
          Number to decide how many iterations of the model is train over the whole train data set
        lr_decay : string
          Decide whether to use learning rate decay in which mode or not.
          "linear" is decreasing the learning rate with subtracting a constant value decay_rate each
          epoch(possible to be negative learning rate).
          "power" is decreasing the learning rate with multiplying a constant value lesser than 1 decay_rate each epoch.
        decay_rate : float
          The decay rate of the learning rate each epoch. Is used if lr_decay is not None.
        weight_decay : float
          Decoupled weight decay parameter when using "adamW" optimizer
        save_model : bool
          Set True to enable saving the best model weights
        save_path : string
          If save_model is True. Path to save the best model weights
        monitor : string
          If save_model is True. Metric to decide the best model weights
        mode : string
          If save_model is True. Decide the 'min' or 'max' of metric value to decide the best model weights

        References
        ----------
        "adamW", https://arxiv.org/abs/1711.05101
        """
        # The learning rate epoch decay
        if lr_decay == 'linear':
            length = len(X_train)
            steps = length / self.batch_size
            boundaries = []
            values = []
            for i in range(epochs):
                boundaries.append(int(steps * (i + 1)))
                values.append(lr - decay_rate * i)
            boundaries.pop()

            lr = PiecewiseConstantDecay(boundaries=boundaries, values=values)
        elif lr_decay == 'power':
            length = len(X_train)
            steps = length / self.batch_size
            boundaries = []
            values = []
            for i in range(epochs):
                boundaries.append(int(steps * (i + 1)))
                values.append(lr * pow(decay_rate, i))
            boundaries.pop()

            lr = PiecewiseConstantDecay(boundaries=boundaries, values=values)

        # Apply optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)
        elif optimizer == 'adamW':
            opt = AdamW(weight_decay=weight_decay, learning_rate=lr)
        else:
            raise ValueError("Optimizer doesn't exists!")

        # Compile model
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        # Train model and record training history
        if save_model:
            # Add save best model weights checkpoint
            model_checkpoint_callback = ModelCheckpoint(
                filepath=save_path,
                save_weights_only=False,
                monitor=monitor,
                mode=mode,
                save_best_only=True)
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, shuffle=True, validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback])
        else:
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, shuffle=True, validation_data=(X_val, y_val))
        self.history = history.history
