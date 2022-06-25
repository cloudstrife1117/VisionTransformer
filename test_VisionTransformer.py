import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from Modules.TransformerModel import VisionTransformer


def main():
    # Load Cifar-10 Dataset to train set and test set
    (_, _), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Directory path of the saved model
    model_name = 'ViT_225_256'
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    model_path = ROOT_PATH + "/Models/" + model_name

    # Define variables and hyperparameters
    batch_size = 20  # Size of input batch

    # Load the saved VisionTransformer model
    ViT = tf.keras.models.load_model(model_path)

    # Calculate and print the test set result
    results = ViT.evaluate(X_test, y_test, batch_size=batch_size)
    print("Test Acc:", str(results[1]*100) + "%")


if __name__ == '__main__':
    main()
