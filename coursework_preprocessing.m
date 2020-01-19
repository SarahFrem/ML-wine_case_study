%-------------------------------------------------------------------------------------------------%
% this file is about the initial data analysis, data preprocessing choices and first figures of the poster
% -------------------------------------------------------------------------------------------------%
clear all;
clc;
%% load data

data_wine = readtable('./initial_data/winequality-red.csv');
var_names = {'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'};
data_wine.Properties.VariableNames = var_names ; 

%% checking for imbalanced data 
cat_names = unique(data_wine.quality);
number_per_categories = arrayfun(@(x) sum(data_wine.quality==x), cat_names);
disp('--------- checking number of observations per category -----------')
disp(number_per_categories)

figure;
bar(cat_names, number_per_categories);
title('number of observations per quality');
xlabel('quality');
ylabel('number of observations');
text(3:8, cat_names, num2str(number_per_categories),'vert','bottom','horiz','center');

%% checking for missing values 
disp('--------- checking missing values -----------')
disp(sum(ismissing(data_wine),1));

%% checking for pearson correlation between variables 
correlation = array2table(corrcoef(data_wine{:,1:end-1}),'RowNames', var_names(:, 1:end-1), 'VariableNames', var_names(:, 1:end-1));
disp('--------- checking pearson correlation between features -----------')
disp(correlation)
figure;
h = heatmap(var_names(:, 1:end-1), var_names(:, 1:end-1), corrcoef(data_wine{:,1:end - 1}));
h.Title = 'Pearson correlation matrix';
h.Colormap = parula;

%% removing some features
%{
based on our domain knowledge and explanation features given with the
dataset: 
- free sulfur dioxide is included in total sulfur dioxide: free sulfur dioxide is
removed
- based on the data exploration : density and ph have the same behaviour
upon quality:
density and ph are removed
- to avoid multicollinearity in independent variables (ordinal regresion
assumption): citric_acid is removed
%}
data_wine = removevars(data_wine,{'free_sulfur_dioxide', 'density', 'pH', 'citric_acid'});
var_names = {'fixed_acidity', 'volatile_acidity', 'residual_sugar', 'chlorides', 'total_sulfur_dioxide', 'sulphates', 'alcohol', 'quality'};

correlation = array2table(corrcoef(data_wine{:,1:end-1}),'RowNames', var_names(:, 1:end-1), 'VariableNames', var_names(:, 1:end-1));
figure;
h = heatmap(var_names(:, 1:end-1), var_names(:, 1:end-1), corrcoef(data_wine{:,1:end - 1}));
h.Title = 'correlation matrix on used features';
h.Colormap = parula;

%% removing outliers 
% the idea is to minimize the noise in the dataset without removing
% essential data points in the poorest classes
% we then use the mahalanobis distance:
%   -based on the correlation matrix per quality
%   -and based on the mean per quality 
%(it's relevant since we don't have the same mean and variance per category)

obj = fitcdiscr(data_wine{:,1:end - 1}, data_wine{:,end});
mean_per_category_and_features = obj.Mu;
covar_per_category_and_features = obj.Sigma;
mahalanobis_distances = mahal(obj, data_wine{:,1:end - 1});
% it gives the matrix of the squared mahalanobis distances from data and the covariance and mean of each classes
mahalanobis_distances_t = array2table([mahalanobis_distances, data_wine{:,end}],'VariableNames', {'cat3','cat4','cat5','cat6','cat7','cat8', 'quality_class'}); 

%we check histograms
sub_5 = mahalanobis_distances_t(mahalanobis_distances_t{:,7}==5,3);
sub_6 = mahalanobis_distances_t(mahalanobis_distances_t{:,7}==6,4);
sub_7 = mahalanobis_distances_t(mahalanobis_distances_t{:,7}==7,5);

figure;
nbis=100;
subplot(3, 1, 1);
hist(sub_5{:,1}, nbis);
title('Histogram of squared mahalanobis distances from quality 5');
subplot(3, 1, 2);
hist(sub_6{:,1}, nbis);
title('Histogram of squared mahalanobis distances from quality 6');
subplot(3, 1, 3);
hist(sub_7{:,1}, nbis);
title('Histogram of squared mahalanobis distances from quality 7');

%decison : 
%based on the histograms we remove rows where dist>=30 for quality=6 and 7
%and rows where d>=40 for quality=5
outliers_detection_cat5 = (mahalanobis_distances_t{:,3}>=40 & mahalanobis_distances_t{:,7}==5);
outliers_detection_cat6 = (mahalanobis_distances_t{:,4}>=30 & mahalanobis_distances_t{:,7}==6);
outliers_detection_cat7 = (mahalanobis_distances_t{:,5}>=30 & mahalanobis_distances_t{:,7}==7); 

outliers_detection = outliers_detection_cat5 | outliers_detection_cat6 | outliers_detection_cat7;
disp('----------outliers detection, number of data removed -----------')
disp(sum(outliers_detection))

% cut
data_wine_cleaned = data_wine(~outliers_detection,:);

number_per_categories_cleaned = arrayfun(@(x) sum(data_wine_cleaned.quality==x), cat_names);
disp('--------- after cleaning : number of observations per category -----------')
disp(number_per_categories_cleaned)


%% data visualization in 2D using tsne 
wine_embedded = tsne(data_wine_cleaned{:,1:end-1},'Algorithm','Exact','Exaggeration', length(cat_names), 'NumDimensions', 2, 'Standardize', true);
figure;
gscatter(wine_embedded(:,1),wine_embedded(:,2),data_wine_cleaned{:,end});
title('2-D Embedding with Tsne')

%since we have an imbalanced classes issue, we will use an oversampling method
% we decide to use bordeline smote oversampling rather than smote because
% we want to keep natural inliers and outliers within each minority class
% SMote method would in our case create fake bridges inside classes 

%% oversampling method using bordeline smote 
% we will use the borderline smote library in python because the only
% matlab source we found is the following one
% source: github repository of @Nekooeimehr based on his research paper http://www.sciencedirect.com/science/article/pii/S0957417415007356
% but the running time is much higher than in python. 





