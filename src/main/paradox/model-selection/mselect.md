# Model Selection

Model selection is the task of choosing a model of a set of candidate models. In this impelementation the candidate models are created by first setting the structure of the model then providing a set of values or ranges for the model hyperparameters. To choose a model there needs to be a way of predicting a model's score based on the hyperparameters. For that we used a technique called Bayesian optimization. Often times bayesian optimization is implemented using the Gaussian Processes but we used a neural network to give us the predictive mean and predictive variance. The other part of bayesian optimization is the acqusition function. The acquisition function helps choose the next candidate model to use based on its predictive mean and predictive variance. 

## Model Selection Configuration
|  Property Name                             |  Meaning              |
| -----------------------------------------: | --------------------- |
|``acquisition-function``                    |The acquisition function to use. |
|``concurrent-evaluations``                  |The number of concurrent evaluations. |
|``score-sample-size``                       |The number of candidate models to be sampled |
|``update-frequency``                        |The number of evaluations that need to happen before the predictive neural network was updated. |
|``total-evals``                             |The total number of evaluations before model selection stopped. |
|``categorical``                             |Set of hyperparameters which are categorical and valid values for the hyperparameter. |
|``real``                                    |Set of hyperparameters which use real values and valid ranges for the hyperparameter. |
|``int``                                     |Set of hyperparameters which use integer values and valid ranges for the hyperparameter. |
