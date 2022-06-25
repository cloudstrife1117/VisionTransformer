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
        model : string
          The parameter to decide which model to use.
          "ViT" is the baseline Vision Transformer classification model.

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
        # Create Inputs
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
        x_norm1 = LayerNormalization(epsilon=1e-6)(x)
        x_mha = MultiHeadAttention(head_size=self.proj_dim, num_heads=self.num_heads, dropout=self.dropout)([x_norm1, x_norm1, x_norm1])
        x_concat1 = Add()([x_mha, x_norm1])
        x_norm2 = LayerNormalization(epsilon=1e-6)(x_concat1)
        x_mlp1 = self.MLP(x_norm2, self.proj_dim)
        x_concat2 = Add()([x_mlp1, x_concat1])

        return x_concat2

    def MLP(self, inputs, units):
        x1 = Dense(units*2, activation=gelu)(inputs)
        x1 = Dropout(self.dropout)(x1)
        x2 = Dense(units, activation=gelu)(x1)
        x2 = Dropout(self.dropout)(x2)

        return x2

    def MLP_head(self, inputs, units):
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

        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)
        elif optimizer == 'adamW':
            opt = AdamW(weight_decay=weight_decay, learning_rate=lr)
        else:
            raise ValueError("Optimizer doesn't exists!")

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

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
