README.md still working in progress, there will be a final report here of the experiment, examples and dependencies. References would also be all added later on.

# Cifar-10 Dataset Classification with Vision Transformer

## Author
**Name:** Jeng-Chung, Lien<br/>
**Email:** masa67890@gmail.com

## Table of Contents
temp

## Introduction
In 2017, when the paper “Attention is All You Need” was introduced, it became a dominant model structure and architecture in the natural language processing(NLP) domain. Due to the component of the multi-head self-attention, an attention mechanism, and the scalability ,the availability of parallel computation and the flexibility of the transformer model. Transformer models have become the state of art model in NLP. Vision Transformer, an image classification model using transformers is implemented in this project[[1]](#reference_anchor1). Vision Transformer is a model created by the google research and brain team. Due to the expensive computation of the transformer model on images they introduced this model to make transformers possible on images, this has opened a new era on images rather than using convolution neural networks. Transformers is a new era of state of art models on multiple data types not limited on images. We will be using this self-implemented Vision Transformer from the original paper to perform some experiment on a small data set for demonstration, Cifar-10, an image classification data set. We then will discuss the components of the model, the hyperparameters, the scale of different data sizes and the inductive bias that was introduced in the paper.

## CIFAR-10 Dataset
The dataset to perform image classification here would be using the Cifar-10 dataset. The Cifar-10 dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. These data are a tiny subset from the 80 million tiny image dataset, which has 10 labeled classes(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). There is no additional pre-processing that is needed on these data, they are all in consistent sizes of 32x32x3. What is done is only using min-max scaling to scale the pixels from the range of 0-255 to 0-1. Here the dataset consists of 60000 images in total, where the dataset is split into 50000 as the train set and 10000 as the test set. Where there are 5000 images per class in the train set and 1000 images per class in the test set. Below are the examples of 10 random images in each class.

**CIFAR-10 Dataset Figure:**<br/>
![CIFAR10_example.png](Figures/Dataset/CIFAR10_example.png)

## Vision Transformer Model
Vision Transformer is an image classification model using transformer architecture, here we will be discussing each component and structure of the vision transformer. Below is the whole structure of the Vision Transformer from the original paper. We will separate this model into two parts for the ease of explanation of the model components. Where the two parts would be the 1. Image Patch + Position Embeddings to feed in the Vision Transformer and the 2. Vision Transformer structure itself.

**Vision Transformer Model Figure:**<br/>
![ViT.png](Figures/ViT/ViT.png)

1. **Image Patch + Position Embeddings**<br/>
For the Image Patch + Position Embeddings we will separate this part into four components for explanation. a. Patches of Images, b. Image Embeddings, c. Class Token, and d. Position Embeddings. The components are shown below.

    <br/>**Embeddings Components Figure:**<br/>
    ![ViT_Embeddings.png](Figures/ViT/ViT_Embeddings.png)
    <br/><br/>
    1.a. **Patches of Images**<br/>
    The patches of images are simply just splitting the images into patches of images. For example for a 16x16 image, if we want patches of size 4x4. We simply split the image into 4 patches of size 4x4 shown in the image below.

    <br/>**Patches Figure:**<br/>
    ![1_a_ImagePatches.png](Figures/ViT/1_a_ImagePatches.png)
    <br/><br/>
    1.b. **Image Embeddings**<br/>
    For image embeddings we simply take the image patches from part 1.a. and pass each of the patches to the same linear layer, where the weights will all be the same when mapping each patch to vectors before updating the weights. This basically is creating a dense layer with linear activation for the model to learn itself of how it would like to map the set of patches from a given dataset to vector space. Basically a subset of pixels group in an image as a vector representation. This process is shown in the below image.

    <br/>**Image Embeddings Figure:**<br/>
    ![1_b_ImageEmbeddings.png](Figures/ViT/1_b_ImageEmbeddings.png)
    <br/><br/>
    1.c. **Class Token**<br/>
    For the class token, it is a learnable vector that represents the whole image. As stated in the Vision Transformer paper, this is similar to the BERT’s class token. The usage for adding this additional class token is to not introduce any bias towards any output of the corresponding patches’ input. Since transformers would output the result with a specific input that views the correlation with other input. Here by creating an additional whole image vector representation to learn on its own and output a class representation vector at the output of the transformer. For a learnable class token it would be taking an item as a one hot vector through a dense layer with linear activation as the representation, shown in the image below.

    <br/>**Class Token Figure:**<br/>
    ![1_c_ClassToken.png](Figures/ViT/1_c_ClassToken.png)
    <br/><br/>
    For not introducing any bias towards the input of patches, below is a visual representation of how the transformer would output given inputs. If only using a single vector of the P1 output as the vector to input to the MLP head it would bias towards patch 1 of the image, hence taking Pn output as the vector to input to the MLP head would bias towards patch n. Therefore by creating an additional class token through the transformer output P0 as a whole image representation to input into the MLP head for classification.

    <br/>**Patch Bias Figure:**<br/>
    ![1_c_ClassTokenBias.png](Figures/ViT/1_c_ClassTokenBias.png)
    <br/><br/>
    1.d. **Position Embeddings**<br/>
    Here for position embeddings, it would be a 1D rastar order learnable position embedding. As stated in the Vision Transformer paper also for the original transformer model, using no position embeddings it would just be a bag of patches or bag of words to the transformer. Since the transformer model itself in nature doesn’t recognize any position perspective, adding additional learnable position embeddings to the embeddings as an input would give the transformer model additional information of position to the data, this would boost the performance of the transformer model. Where in this case is different from the original transformer model where they use different frequencies of wave functions as positional representations, here we let the model learn the positional order itself. Basically, it would be using one-hot vector representations of p+1 positions passing through a dense layer with linear activation to get the positional embeddings, where there are p patches and 1 class token, the image shown below. From a high level, we could imagine this as a puzzle, where there are patches and the model would figure out how to construct these patches into a whole image. The reason for not using a 2D representation which in nature makes sense for 2D structure for images, is due to there is no boost in performance stated in the Vision Transformer paper.

    <br/>**1D Learnable Position Embeddings Figure:**<br/>
    ![1_d_PositionEmbedding.png](Figures/ViT/1_d_PositionEmbedding.png)
    <br/><br/>
    For the whole process of part 1. to generate the image patch + position embeddings, taking all the vector representations of each patch of a single image as the image embeddings in 1.b(p vectors with size n), we concat this with the class token(single vector size n) to get a p+1 vector with size n. We then take this p+1 vector with size n and do addition with the position embeddings(p+1 vectors with size n) to get the final embeddings(p+1 vectors with size n), which are the patches of the image vector representation with positional information with an addition class token of the whole image representation. This process is shown in the image below. We then will take these final embeddings as the input to the transformer encoder.

    <br/>**Process Overview of Part 1 Figure:**<br/>
    ![1_FinalEmbedding.png](Figures/ViT/1_FinalEmbedding.png)
    <br/><br/>

2. **Vision Transformer Structure**<br/>
For the Vision Transformer Structure we will separate this part into two components for explanation. a. Transformer Encoder, and b. MLP Head. The components are shown below.

    <br/>**Vision Transformer Structure Figure:**<br/>
    ![ViT_VisionTransformerStructure.png](Figures/ViT/ViT_VisionTransformerStructure.png)
    <br/><br/>
    2.a. **Transformer Encoder**<br/>

    <br/>**Transformer Encoder Figure:**<br/>
    ![2_a_TransformerEncoder.png](Figures/ViT/2_a_TransformerEncoder.png)
    <br/><br/>

    * **Description:**
    The above image is the structure of the transformer encoder for vision transformer. Different from the encoder of the original transformer paper, they apply the normalization before each component. Hence the structure here is in the sequence of i. layer normalization(Norm), ii. multi-head attention, iii. layer normalization(Norm), and iv. multilayer perceptron(MLP). Where there are two small skip connections, the first one is taking the input of i. and adding it with the output of ii. to pass it as the input of iii. The second one is taking the input of iii. and adding it with the output of iv. as the output of the transformer encoder block. We could stack up these transformer encoder blocks to add the depth and complexity of the model, the Lx here means the parameter of the number of stacks of these transformer encoder blocks. To note that there are dropout layers in between components through the network.
    * **Components:**
        * **Multi-Head Attention:**
        This is the main component of the transformer model, an attention mechanism. From a high level perspective taking an image as an example, it is taking the image itself and searching on itself to see which parts are the important parts to focus and take account of the information, hence putting attention on specific parts of the image. For example, if there is a classification task of classifying human eye colors from a whole human face image, the attention would focus on the parts of the eye mainly.
        <br/><br/>**Multi-Head Attention Figure:**<br/>
        ![2_a_MultiHeadAttention.png](Figures/ViT/2_a_MultiHeadAttention.png)
        <br/><br/>
        Above is the structure of the Multi-Head Attention(Right.) and the structure of the Scaled Dot-Product in Multi-Head Attention(Left.). 
            * **Scaled Dot-Product Attention:**
            By first explaining the Scaled Dot-Product Attention, it is taking multiple vectors into a matrix as a query Q, multiple vectors into a matrix as key K and multiple vectors into a matrix as value V. Here the QKV are all taken from the same input passing through different dense layers with linear activation, here the query is matmul with the key then scaled down with the value of the square root of the dimension of the key. We could ignore the Mask in this part since it is used in the decoder of the original transformer paper. This scaled down matrix is then passed through a softmax, in a form of softmax through each row, this forms a probability or a correlation matrix where the focus and attention occurs. This correlation matrix is then matmul with the value matrix to form the attention score matrix. The formula is shown below.
            <br/><br/>**Scaled Dot-Product Attention Formula Figure:**<br/>
            ![2_a_AttentionFormula.png](Figures/ViT/2_a_AttentionFormula.png)
            <br/><br/>
            * **Multi-Head Attention:**
            Multi-Head Attention: For multi-head attention, we are taking h different scaled dot-product attention where all of these h attention heads will have different dense layers with linear activation. The output of these attention heads are then concat together and passed through another dense layer with linear activation to reduce the dimension. From a high level perspective of what the multi-head attention is doing, we could look at the human eye color example previously. If we use two attention heads in the multi-head attention, it is possible that one attention would focus on the eyes in the image and the other attention would focus on the skin color of the image. Using multiple attentions means considering more different features and perspectives that have correlations to focus on and learn.
        * **Layer Normalization:**
        For layer normalization, different from batch normalization it doesn’t scale individual features through a batch distribution. It normalizes through a single sample with all of the different features.
        * **MLP:** 
        For the multilayer perceptron here it is simply using two dense layers with gaussian-error linear unit(GELU) activation.

    <br/>2.b. **MLP Head**<br/>
    Here the MLP Head is taking only the output corresponding to the class token as input only to consider the whole image representation without any bias towards any of the patches. The components of the MLP Head are simply using a dense layer with GELU activation then through a dense layer with sigmoid or softmax activation for classification.

## Experiment
temp

## Reference
<a name="reference_anchor1"></a>[1] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” Jun. 2021. [Online]. Available: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

## Other Projects
1. ISICs Dataset Segmentation with Improved UNet. [Github Link](https://github.com/shakes76/PatternFlow/tree/topic-recognition/recognition/Segmentation-ISICs-ImprovedUnet-s4623205)