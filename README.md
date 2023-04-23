# ambitious_explorations_privacyattacks

# Membership Inference Attack
 

Since Shokri’s membership inference attack is the first of its kind and the most fundamental, it paved way for other attacks as well. A typical model may evolve its weights over training phase, and during inference the model is able to deal with real world data to produce outputs. 
In black box setting, an adversary attacks this model by observing outputs of the model for given input, and adding this output vectors to an attack model’s(which can be arbitrary machine learning model) input, that classifies whether that input was in dataset or not.
In this course, the shadow model comes into play to produce training dataset for the attack model, so that attack model learns to differentiates in-member output vectors from out-member output vectors.
In white box setting, the shadow model holds the same architecture as the target model.

Remembrance Metric
 
All the zoo of membership inference attack models work on one principle, i.e. To infer the presence of information associated to a particular dataset in a private dataset, by only observing the model. This is illustrated in the above figure. Clearly, since a model can hold as much information as the original dataset in the most ideal case, practically evaluating the success of membership inference attack for a given model and expecting nearly 100 percent accuracy is not reasonable. That is so, because a practical model does not overfit on a private dataset, but rigorously regularized before deployment.
In this course, all available attack metrics tend to neglect this fact, and it leads to three main problems:
-It does not allow the benchmark attack models to be properly be evaluated, allowing a risk of the same attack to be more accurate at the hands of adversary
- It does not allow to properly calibrate the hyperparameters of the attack model.
- Also, since this remembrance acts as the ultimate leverage of any attack model, therefore we need to estimate attack metrics of whole zoo of MIA models leading to computational overhead.
Therefore, we propose a following methodology to estimate the remembrance of original target models.
 
Figure: Framework for Target Model’s remembrance

Attack Model
Since attack model is the fundamental part of all membership inference attacks, therefore, its structure needs to be clearly defined that should be invariant to any given target model, and any given attack zoo. In this regard, we also proposed a new architecture for attack model, that would remain same for any model and attack considered.


Figure: Pipeline for attack Model

Here, the temporal distribution of output vector of a target model (which is a probability distribution) via cubic spline interpolation and a non-linear transformation, that would be learned by an attention model to better learn short-term vs long-term dependencies to predict the probability of corresponding input whether in being or out of the given dataset. We consider the following non-linear transformation over a given sequence ‘x’ in our approach.
 
Where ‘A’ is appropriately chosen scalar in above equation. We show the effectiveness of this non-linear transformation, by performing 1000 arbitrary linear transformations over the raw probability vector and the nonlinear transformed probability vector. It is abundantly clear that standard deviation of the nonlinear transformed vector is significantly higher as compared to raw probability vector one, and thus can easily separated by hyperplane (through activation function) leading to overall greater accuracy.

 

 The attack model comprises of multi-headed attention, followed by a 1D-Convolution layer and a single layer perceptron (SLP) to predict of probability of membership. This architecture is invariant with respect to type of given model and attack consideration.



Performance over NLP dataset
For NLP attack consideration, we took the fine-grained cyberbully classification dataset (https://ieee-dataport.org/open-access/fine-grained-balanced-cyberbullying-dataset). This dataset comprises of several Twitter tweets with corresponding different cyberbully class labels. We tend to implore the membership inference attack over this dataset. We considered following models in our membership inference attack consideration. With in the attack model, we integrated our proposed pipeline. The target model is a Transformer architecture which is one of the state of the art text classification model considered in cyberbullying literature.
 
Figure: Models considered in formulation of Membership inference attack over NLP dataset

We also employed our proposed remembrance metric pipeline inorder to compare the original attack metrics with remembrance attack metrics to produce a baseline. We also explored the effect of overlap between training and shadow datasets to determine the robustness of attack. We plotted the results in figure. It is vivid that the attack performance is greatly robust to amount of information about dataset that the attacker knows about the original dataset. Furthermore, the remembrance metrics correlates well with the attack accuracy metrics.
 
Figure: Attack and Remembrance Metrics corresponding  Transformer Model
In order to empirically show that overfitting directly correlates with any membership inference attack, we compared the rememberence metrics over fitted and overfitted target model. The results are shown in figure
 
Figure: Comparison of remembrance metrics over fitted and overfitted target model
In above figure, the graphs show the training and validation loss and accuracy curves over epochs, which clearly signify whether the model is fitting or overfitting. Since the overfitted model has significant gap between training and validation loss curves, with validation loss slightly increasing, therefore it is overfitting. Last, but not least, the remembrance model’s metrics of overfitted model are nearly 13 percent greater than that of fitted model, thus establishing the link.
