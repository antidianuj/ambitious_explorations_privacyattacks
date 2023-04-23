# ambitious_explorations_privacyattacks
## Literature Comparison between Membership Inference Attack and Model Inversion

![image](https://user-images.githubusercontent.com/47445756/233863325-1a80debb-dfce-4502-862c-6f9dffda66d5.png)

## Membership Inference Attack
 
 ![image](https://user-images.githubusercontent.com/47445756/233847698-ad868570-f022-467a-8a16-73fe7cfc61cb.png)


Since Shokri’s membership inference attack is the first of its kind and the most fundamental, it paved way for other attacks as well. A typical model may evolve its weights over training phase, and during inference the model is able to deal with real world data to produce outputs. 
In black box setting, an adversary attacks this model by observing outputs of the model for given input, and adding this output vectors to an attack model’s(which can be arbitrary machine learning model) input, that classifies whether that input was in dataset or not.
In this course, the shadow model comes into play to produce training dataset for the attack model, so that attack model learns to differentiates in-member output vectors from out-member output vectors.
In white box setting, the shadow model holds the same architecture as the target model.

### Remembrance Metric

![image](https://user-images.githubusercontent.com/47445756/233847720-b2982af8-9598-428d-b1b6-456d209234a6.png)

 
All the zoo of membership inference attack models work on one principle, i.e. To infer the presence of information associated to a particular dataset in a private dataset, by only observing the model. This is illustrated in the above figure. Clearly, since a model can hold as much information as the original dataset in the most ideal case, practically evaluating the success of membership inference attack for a given model and expecting nearly 100 percent accuracy is not reasonable. That is so, because a practical model does not overfit on a private dataset, but rigorously regularized before deployment.
In this course, all available attack metrics tend to neglect this fact, and it leads to three main problems:
-It does not allow the benchmark attack models to be properly be evaluated, allowing a risk of the same attack to be more accurate at the hands of adversary
- It does not allow to properly calibrate the hyperparameters of the attack model.
- Also, since this remembrance acts as the ultimate leverage of any attack model, therefore we need to estimate attack metrics of whole zoo of MIA models leading to computational overhead.
Therefore, we propose a following methodology to estimate the remembrance of original target models.

![image](https://user-images.githubusercontent.com/47445756/233847728-d70e716d-13d6-42a0-9636-44a8940a97ea.png)
Figure: Framework for Target Model’s remembrance

Attack Model
Since attack model is the fundamental part of all membership inference attacks, therefore, its structure needs to be clearly defined that should be invariant to any given target model, and any given attack zoo. In this regard, we also proposed a new architecture for attack model, that would remain same for any model and attack considered.

![image](https://user-images.githubusercontent.com/47445756/233847736-63141ba6-281b-4244-8b17-a490c160f9dc.png)
Figure: Pipeline for attack Model

Here, the temporal distribution of output vector of a target model (which is a probability distribution) via cubic spline interpolation and a non-linear transformation, that would be learned by an attention model to better learn short-term vs long-term dependencies to predict the probability of corresponding input whether in being or out of the given dataset. We consider the following non-linear transformation over a given sequence ‘x’ in our approach.
 
![image](https://user-images.githubusercontent.com/47445756/233847750-e736c54f-2b3c-4f31-b420-a6d72828798c.png)

Where ‘A’ is appropriately chosen scalar in above equation. We show the effectiveness of this non-linear transformation, by performing 1000 arbitrary linear transformations over the raw probability vector and the nonlinear transformed probability vector. It is abundantly clear that standard deviation of the nonlinear transformed vector is significantly higher as compared to raw probability vector one, and thus can easily separated by hyperplane (through activation function) leading to overall greater accuracy.

 ![image](https://user-images.githubusercontent.com/47445756/233847754-b091acbf-e97e-45c5-83bf-d17605c697a7.png)

The attack model comprises of multi-headed attention, followed by a 1D-Convolution layer and a single layer perceptron (SLP) to predict of probability of membership. This architecture is invariant with respect to type of given model and attack consideration.



### Performance over NLP dataset
For NLP attack consideration, we took the fine-grained cyberbully classification dataset (https://ieee-dataport.org/open-access/fine-grained-balanced-cyberbullying-dataset). This dataset comprises of several Twitter tweets with corresponding different cyberbully class labels. We tend to implore the membership inference attack over this dataset. We considered following models in our membership inference attack consideration. With in the attack model, we integrated our proposed pipeline. The target model is a Transformer architecture which is one of the state of the art text classification model considered in cyberbullying literature.
 
 ![image](https://user-images.githubusercontent.com/47445756/233847769-0c862428-feef-4862-9926-3effbd867035.png)
Figure: Models considered in formulation of Membership inference attack over NLP dataset

We also employed our proposed remembrance metric pipeline inorder to compare the original attack metrics with remembrance attack metrics to produce a baseline. We also explored the effect of overlap between training and shadow datasets to determine the robustness of attack. We plotted the results in figure. It is vivid that the attack performance is greatly robust to amount of information about dataset that the attacker knows about the original dataset. Furthermore, the remembrance metrics correlates well with the attack accuracy metrics.
 
![image](https://user-images.githubusercontent.com/47445756/233847807-dc102579-0aa2-4730-acf7-2fd1da1bf0c2.png)
Figure: Attack and Remembrance Metrics corresponding  Transformer Model

In order to empirically show that overfitting directly correlates with any membership inference attack, we compared the rememberence metrics over fitted and overfitted target model. The results are shown in figure
 
![image](https://user-images.githubusercontent.com/47445756/233847825-74cb7941-0f5f-417a-9a29-4ab347e2d8a9.png)
Figure: Comparison of remembrance metrics over fitted and overfitted target model

In above figure, the graphs show the training and validation loss and accuracy curves over epochs, which clearly signify whether the model is fitting or overfitting. Since the overfitted model has significant gap between training and validation loss curves, with validation loss slightly increasing, therefore it is overfitting. Last, but not least, the remembrance model’s metrics of overfitted model are nearly 13 percent greater than that of fitted model, thus establishing the link.


### Performance over Computer Vision dataset
We also considered several image classification datasets and target models. The attack summary is provided in following table.

| **Dataset**         | **Model**                                                                  | **Attack Model** | **Shadow Model**                                                           | **At-P** | **At-R** | **At-F** | **Re-P** | **Re-R** | **Re-F** |
| ------------------- | -------------------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | -------- |
| CIFAR-10            | 0.5 million parameter CNN                                                  | Attention Model  | 0.5 million parameter CNN                                                  | 0.5604   | 0.7964   | 0.6361   | 0.5496   | 0.3928   | 0.4582   |
| CSE7512-00 (Kaggle) | 0.5 million parameter CNN                                                  | Attention Model  | 20 million parameter Xception model (Imagenet pretrained-transfer learned) | 0.5044   | 0.2038   | 0.2907   | 0.6237   | 0.067    | 0.1218   |
| CSE7512-00 (Kaggle) | 20 million parameter Xception model (Imagenet pretrained-transfer learned) | Attention Model  | 20 million parameter Xception model (Imagenet pretrained-transfer learned) | 0.6390   | 0.8768   | 0.6839   | 0.5907   | 0.164    | 0.2568   |




## Model Inversion Attack
### Gradient Descent based Attack
I reconstructed the most fundamental model inversion attack can be be model as a standard gradient descent based minimization problem. This attack is demonstrated in following figure. Here C(1,f(x’)) is the cost function corresponding to desired label and the model’s output f(x’).
 
![image](https://user-images.githubusercontent.com/47445756/233848003-f8c65286-ecaf-4efe-86c6-c3df02a3d7fe.png)
Figure: Model Inversion Attack Pipeline


We employed the model inversion attack over AT&T Database of Faces , and the target model is a two layer deep multi-layer perceptron. Following is the result over 100 epochs of model inversion attack.

 ![image](https://user-images.githubusercontent.com/47445756/233848020-c1ecbc85-d02c-42b2-8522-3dffd3574bf3.png)
Figure: Results of Model Inversion Attack

### LASSO Based Attack
#### Lasso Regression
We also performed model inversion attack by fitting a LASSO regression model over synthetic samples similar to training data, and the output probability vectors of the target model corresponding to the synthetic data.  This essentially forms a black box attack. Following are the results.
![image](https://user-images.githubusercontent.com/47445756/233848037-fa7e7b08-1dfa-4f55-95a9-76fc38a4884e.png)
 
#### PCA-Normalized Lasso Regression
In order to reduce the computation complexity associated with the above process and also eliminating the need for further post processing to improve the quality of the reconstructed images, we first perform PCA over the probability vectors to reduce its dimensionality and aso improve its feature representation. We also employed normalization of input and output data of the LASSO model, to improve the reconstruction accuracy. As the consequence, following are the results.
 ![image](https://user-images.githubusercontent.com/47445756/233848050-9165f814-1599-4aae-8228-a45ad0f76110.png)

### GAN Based Attack
We also employed GAN based model inversion attack in a white box setting, where utilize the target model as the discriminator model, and the generator model comprises of transposed-convolution architecture. The dataset used would be synthetic image samples similar to the target model. The input to the generator is essentially the predicted probability vector (by the original target model) and the output of discriminator are hard labels. The results of this attack are as follows:
![image](https://user-images.githubusercontent.com/47445756/233848057-a4006c12-d0eb-4b9e-8819-86a4957f545c.png)


