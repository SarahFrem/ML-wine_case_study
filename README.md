# Machine learning, first study case on UCI Wine Quality Data Set.
Quality can either be considered as a class or an ordinal ranked target. Therefore two different algorithms are fitted and compared: Random Forest as multi-classification algorithm as well as an Ordinal Logistic Regression.

## Initial data analysis including basic statistics
Data Set containing 1599 observations, 11 chemical compounds features and 1 ranked target, the wine quality divided in 6 classes from 3 to 8.
Initial data analysis shows that :
  - Clear imbalanced quality classes: 3 minority classes containing 10 , 18 and 53 samples versus 3 majority classes having 681, 638 and 199 observations.
  - Some initial features are quite high correlated based on Pearson computation.
  - Features per class have different trends and densities revealing that they don’t contribute proportionately to the target output.
  - All classes contain clearly outliers.
  
 ![correlation](corr_features.jpg | width=100 )
 
 ## Data preprocessing
 In order to avoid multicollinearity within independent variables and to keep meaningful features based on domain knowledge and densities:    
 - Citric acid, pH, density and free sulfur dioxide were removed. 
 - Outliers in the densest quality classes 5, 6 and 7 were removed based on Mahalanobis distance computed in each class separately.
   
 ## Brief summary of the machine learning models
 
#### Random Forest Classification:
Supervised ensemble learning algorithm for classification.
Based on improving single Decision Tree algorithm by constructing a multitude of trees and outputting the decision class from the mode.
-- Pros 
Powerful and accurate results, good and robust performance on many problems such imbalanced data and even nonlinear problems.
No formal distributional assumptions are needed neither data preparation. Random Forest is non-parametric and therefore, can handle skewed and multimodal data as well as categorical data and ordinal. 
Reduce over-fitting of a single decision tree.
The out of bags error gives a truly good idea of how well the model would fit on a test set and allows the training on the full initial dataset. 
-- Cons 
Less interpretability of the model. 
Over-fitting can easily occur, hyper parameters need to be tuned carefully. 
Random Forest prediction process is time inefficient than other algorithms.

#### Ordinal Logistic Regression:
Supervised machine learning algorithm for ordering or ranking patterns. 
Ordinal Regression is a member of the family of regression analyses where the dependent variable is ordinal. 
The proportional odds model states: 𝑙𝑜𝑔𝑖𝑡 𝑃 𝑌 ≤ 𝑗 = 𝛼𝑗 − σ𝛽𝑘𝑋𝑘 where j are the levels of ordered categories and β is the common slope among categories.
-- Pros 
Optimized for ordinal data. 
Independent variables can be ordinal, categorical and continuous, giving to the model a powerful flexibility. 
It helps to keep an observed natural order in the classes instead of treating them naively independent.
-- Cons 
Requires some assumptions that are usually violated such as no-multicollinearity within features and proportional odds. 
Fits only certain types of datasets, cannot be generalized on any dataset. 
Model less known and interpretability can be less easy.

## Description of choice of training and evaluation
Training choices: Used a stratified 5-fold on the initial dataset. This is done because a very few samples is available in the poorest class. These folds are used to evaluate each model, find the best hyperparameters and to analyse scores among folds and classes. Each fold is scaled, and the three lowest classes are oversampled through Borderline Smote with 3 KNN and a multiplication coefficient from 2.5 to 5.5. Borderline Smote had been chosen in order to keep natural inliers and outliers within these classes. 

Evaluation method: Regarding specifically Random Forest, we check the error curve of training folds and validation folds to prevent overfitting and to find the required number of trees needed for the model to learn. For each fold and each class in both models, we compute the recall, precision and F1 score. Since we have an imbalanced dataset, these scores are then weighted by the number of class support in each fold. We use the average of these scores over the 5 folds in order to have a final accuracy and to pick the best hyperparameters. Finally a final split is made on the whole dataset in order to get a test accuracy score.

## Final results
Results prove that Random Forest Classification clearly outperforms Ordinal Regression on training and validation. Although Ordinal Regression manages to reach, likely by luck, 50% False-Positive regarding the poorest class 3 in the final test, Random Forest still gives a better weighted score.
Random Forest is known to be robust against imbalanced classes. Hence, treating this issue first improves predictions rather than considering natural ranking order in qualities.
Over 5 folds, Ordinal Regression does not fit well on the training set, the deviance being around 3.103 . First data analysis shows that features don’t contribute proportionally to classes. Therefore, low Ordinal Regression scores are likely due to the strong assumption of proportional odds. • Regarding Random Forest, training scores show that the model did not overfit and contribute to good weighted scores on average over validation and test stages. However, looking more closely on each class highlights that the introduction of Smote Borderline into the training subset has slightly improved the performance of minority classes but does not prevent it of overfitting.
In the pre-processing phase, outliers in minority classes were not removed to prevent information loss. However, any available minority outlier will have a significant effect on the model performance. This explains why in Random Forest, class 3 turns out to be the hardest minority class to predict over 5 folds. 
Finally, Random Forest offers to the alcohol and volatile acidity compounds the highest importance and fixed acidity, residual sugar and chlorides, the lowest one. Ordinal Regression supplies a slight difference: volatile acidity remains to be the highest proportional odds coefficient on each fold whereas alcohol seems to be among the less relevant. Furthermore, total sulfur dioxide is never significant based on its p-value superior at 5%.
   

