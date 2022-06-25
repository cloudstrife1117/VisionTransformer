import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from Modules.TransformerModel import VisionTransformer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def main():
    # Load Cifar-10 Dataset to train set and test set
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Print the shape of train and test set
    print("x_train Shape:" + str(X_train.shape))
    print("y_train Shape:" + str(y_train.shape))
    print("x_test Shape:" + str(X_test.shape))
    print("y_test Shape:" + str(y_test.shape))

    # Define variables and hyper-parameters
    # Input image settings and image patches settings
    image_size = 32  # The width and height of the image
    channels = 3  # The number of the channels of the image, eg. RGB is 3 channels
    patch_size = 4  # The width and height of the patch
    stride_num = patch_size  # The stride of dividing the image into patches
    rc_patch_num = (image_size - patch_size) // stride_num + 1  # Number of patches of each row or column
    patch_num = rc_patch_num ** 2  # Number of total patches
    # Model complexity settings
    proj_dim = 32  # The dimension size for each patches to project to
    num_heads = 8  # The number of attention heads in Multi-head self-attention
    stack_num = 6  # The number of transformer encoder stacks(depth of the model)
    dropout = 0.1  # The dropout probability trough the whole model
    classes = 10  # The number of output classes
    # Training settings
    optimizer = 'adamW'  # Optimizer, eg. 'adam', 'adamW'
    weight_decay = 0.00001  # The weight decay rate for the Adam Decouple Weight Decay
    loss_function = SparseCategoricalCrossentropy(from_logits=True)  # The loss function
    metrics = [SparseCategoricalAccuracy(name='Acc')]  # Evaluation metrics
    batch_size = 20  # Size of training batch
    learning_rate = 0.001  # The number of the initial learning rate
    epochs = 50  # The total epochs to train on
    lr_decay_mode = 'linear'  # Learning rate mode, eg. 'linear', 'power', None for no changes
    decay_rate = 0.0000199  # The learning rate decay value trough each epoch
    # Save model callbacks settings when training
    is_savemode = True  # Set True to save model trough training
    monitor_metric = 'val_Acc'  # Select the metric name to monitor for saving
    monitor_metric_objective = 'max'  # Select to save the model at the max or min value of the metric trough epochs
    # Directory path to save model weights
    model_name = "ViT_" + str(patch_num) + "_" + str(proj_dim)
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    save_path = ROOT_PATH + "/Models/" + model_name

    # Print model name and number of patches
    print("Training " + model_name)
    print("Number of Patches: " + str(patch_num))

    # Define the VisionTransformer Model
    ViT = VisionTransformer(input_shape=(image_size, image_size, channels),
                            batch_size=batch_size,
                            classes=classes,
                            patch_size=patch_size,
                            stride_num=stride_num,
                            patch_num=patch_num,
                            proj_dim=proj_dim,
                            num_heads=num_heads,
                            stack_num=stack_num,
                            dropout=dropout
                            )

    # Print the model details
    ViT.summary()

    # Train and save the model
    ViT.train(X_train=X_train,
              X_val=X_test,
              y_train=y_train,
              y_val=y_test,
              optimizer=optimizer,
              lr=learning_rate,
              loss=loss_function,
              metrics=metrics,
              epochs=epochs,
              lr_decay=lr_decay_mode,
              decay_rate=decay_rate,
              weight_decay=weight_decay,
              save_model=is_savemode,
              save_path=save_path,
              monitor=monitor_metric,
              mode=monitor_metric_objective
              )


if __name__ == '__main__':
    main()
