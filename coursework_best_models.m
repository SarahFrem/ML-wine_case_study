%--------------------------------------------------------------------------------------------------------------------------------%
% file containing the whole project, loading data, splitting, data preprocessing, the 2 best models, scores, results and figures
% -------------------------------------------------------------------------------------------------------------------------------%

clear all;
clc;

%% ---- loading initial data -----------

data_wine = readtable('./initial_data/winequality-red.csv');


%% ------- preprocessing data ------------ 

%we launch preprocess_data.m that does in a function what we found out in
%data analysis
%for more explanation and data visualisation, please run coursework_preprocessing.m

[cleaned_wine, var_names] = preprocess_data(data_wine);
cleaned_wine.Properties.VariableNames = var_names;

%% ------ KFold splitting and checking ---------
% Since we are running out of samples in poorest classes (10 in class 3 and 18 in class 8):
% we first split our whole initial dataset into 5 different stratified folds to perform a cross validation 

%we shuffle the dataset before cv partitioning
cleaned_wine = cleaned_wine(randperm(size(cleaned_wine, 1)), :); 

%we split in 5 folds the initial shuffled dataset in a statrify way to keep enough data per class
cv_split = cvpartition(cleaned_wine.quality,'KFold',5,'Stratify',true); 

% we loop over each fold and we check how many observations per category we have 
for i = 1:cv_split.NumTestSets
    training_folds = cleaned_wine(cv_split.training(i),:);
    test_folds = cleaned_wine(cv_split.test(i),:);

    disp(['--------- Training Fold number ', num2str(i),'---------'])
    nb_categories_fold_training = arrayfun(@(x) sum(training_folds.quality==x), unique(training_folds.quality));
    disp(nb_categories_fold_training)
    
    disp(['--------- Testing Fold number ', num2str(i),'---------'])
    nb_categories_fold_test = arrayfun(@(x) sum(test_folds.quality==x), unique(test_folds.quality));
    disp(nb_categories_fold_test)
    
    disp('-------------------------------------------------')
end

% we now have our 5 different folds, on which we can perform our ML models within the loop.
% Before so we must apply: 
% 1. normalizing (substracting mean and dividing by std)
% 2. oversampling from borderline SMOTE ()

% NB: normalization process will be done in matlab whereas oversampling is
% done in python since no borderline SMOTE nad been implemented in matlab


%% ------ normalizing --------
% we apply : z = (x - u) / s where u is the mean and s standard deviation

for i = 1:cv_split.NumTestSets
    training_folds = cleaned_wine(cv_split.training(i),:);
    test_folds = cleaned_wine(cv_split.test(i),:);
    
    % ------ fit Standardization on training_folds ------------
    % (getting mean and std per feature
    
    mean_ = mean(training_folds{:,1:end-1});
    std_ = std(training_folds{:,1:end-1});
    
    % ---------- transform training_folds into Standardized --------
    
    training_folds_scaled = (training_folds{:,1:end-1} - mean_) ./ std_;
    % store it in a table
    training_folds_scaled = array2table([training_folds_scaled, training_folds.quality],'VariableNames',training_folds.Properties.VariableNames);
    
    % -------- transform test_folds into Standardized based on training fit -------- 
    
    test_folds_scaled =  (test_folds{:,1:end-1} - mean_) ./ std_;
    % store it in a table
    test_folds_scaled = array2table([test_folds_scaled, test_folds.quality],'VariableNames',test_folds.Properties.VariableNames);
    
    % ---------------------------------------------------%
    % save the fold in a .txt file in order to apply borderlineSMOTE trhough
    % python exceptionnaly 
    % uncomment the following two lines if needed 

    %writetable(training_folds_scaled, strcat('./scaled_training_folds/scaledFold_train_', int2str(i), '.txt'), 'Delimiter',' ');
    %writetable(test_folds_scaled, strcat('./scaled_testing_folds/scaledFold_test_', int2str(i), '.txt'), 'Delimiter',' ');
   
end 

%% oversampling with borderline SMOTE
% made through python library on the folds saved in the previous section 
% here we load the new folds filled with artificial data 
% information about borderline SMOTE : classes 3 8 and 4 has been resampled
% by multiplying their number of obs by 5.5, 4.5 and 2.5 respectively.
% we use 3 KNN in order to resample


%sm_ = BorderlineSMOTE(
%                k_neighbors=3,
%                m_neighbors=20,
%                sampling_strategy={ 3 : int(df.groupby('quality').size()[3]*5.5),
%                                    8 : int(df.groupby('quality').size()[8]*4.5),
%                                    4 : int(df.groupby('quality').size()[4]*2.5)})

%X_res, y_res = sm_.fit_resample(df[features], df['quality'].values.flatten())

%% -------------------------------------------
% MODEL 1 : RANDOM FOREST CLASSIFICATION
% -------------------------------------------

% we apply our first ML algorithm in each fold in order to do a cross validation 

% we will save in those variables some outputs of the model to plot them
% later on
RF_oob_errors_train = {cv_split.NumTestSets,1};
RF_errors_train = {cv_split.NumTestSets,1};
RF_errors_test = {cv_split.NumTestSets,1};
RF_oob_permuted_predictor_delta_error = {cv_split.NumTestSets,1};

RF_precision_train = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));
RF_recall_train = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));
RF_fscore_train = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));

RF_precision_test = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));
RF_recall_test = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));
RF_fscore_test = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));

RF_weighted_precision_train = zeros(cv_split.NumTestSets, 1);
RF_weighted_recall_train = zeros(cv_split.NumTestSets, 1);
RF_weighted_fscore_train = zeros(cv_split.NumTestSets, 1);
RF_weighted_precision_test = zeros(cv_split.NumTestSets, 1);
RF_weighted_recall_test = zeros(cv_split.NumTestSets, 1);
RF_weighted_fscore_test = zeros(cv_split.NumTestSets, 1);

for i = 1:cv_split.NumTestSets
    
    % ------ reading each fold ------------
    training_set = readtable(strcat(['./oversampled_data/over_train_',int2str(i),'.txt']), 'Delimiter',' ');
    test_set = readtable(strcat(['./scaled_testing_folds/scaledFold_test_',int2str(i),'.txt']), 'Delimiter',' ');
    
    % ------ fitting random forest classifier from TreeBagger ------------
    
    % we control the depth of the tree with MinLeafSize, MaxNumSplits
    clf_rf = TreeBagger(100, ...
        training_set, 'quality', ...
        'Method','classification',...
        'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on', ...
        'Prior', 'Empirical', ...
        'MinLeafSize', 1, ...
        'MaxNumSplits', 2000);
    
    
    % ------ predicting ------------
    predicted_qualities_on_training = clf_rf.oobPredict;
    predicted_qualities_on_test = clf_rf.predict(test_set);
    
    
    % ------ saving metrics ------------
    
    % out of bags misclassification probability curve on training
    RF_oob_errors_train{i} = oobError(clf_rf, 'Mode', 'Cumulative'); 
    
    % errors train
    RF_errors_train{i} = error(clf_rf, training_set); 
    
    % errors test
    RF_errors_test{i} = error(clf_rf, test_set);
    
    % Out-of-Bag Feature Importance (from DeltaError attribute)
    RF_oob_permuted_predictor_delta_error{i} = clf_rf.OOBPermutedPredictorDeltaError;
    
    % ------ confusion matrix and scores per class ------------
    
    % we display for each round confusion matrix and we compute some scores:
    % precision , recall and fscore
    % computed in compute_confusion_matrix.m file
    
    beta = 1; %parameter of fscore
    confusion_matrix_visible = 'off'; %change by 'on' if we want to show confusion matrix 
    [RF_precision_train(i,:), RF_recall_train(i,:), RF_fscore_train(i,:)] = compute_confusion_matrix(clf_rf.Y, predicted_qualities_on_training, strcat({'round '},{int2str(i)},{' RF : confusion matrix on train data'}), beta, confusion_matrix_visible);
    [RF_precision_test(i,:), RF_recall_test(i,:), RF_fscore_test(i,:)] = compute_confusion_matrix(cellstr(string(test_set.quality)), predicted_qualities_on_test, strcat({'round '},{int2str(i)},{' RF : confusion matrix on test data'}), beta, confusion_matrix_visible);

    % ----- weighted scores ------
    
    support_train = arrayfun(@(x) sum(training_set.quality==x), unique(training_set.quality));
    support_test = arrayfun(@(x) sum(test_set.quality==x), unique(test_set.quality));
    
    RF_weighted_precision_train(i) = RF_precision_train(i,:)*support_train/sum(support_train);
    RF_weighted_recall_train(i) = RF_recall_train(i,:)*support_train/sum(support_train);
    RF_weighted_fscore_train(i) = RF_fscore_train(i,:)*support_train/sum(support_train);
    RF_weighted_precision_test(i) = RF_precision_test(i,:)*support_test/sum(support_test);
    RF_weighted_recall_test(i) = RF_recall_test(i,:)*support_test/sum(support_test);
    RF_weighted_fscore_test(i) = RF_fscore_test(i,:)*support_test/sum(support_test);
end

%% -------- RANDOM FOREST - RESULTS  --------

disp('-------- RESULTS BELOW ARE FROM RANDOM FOREST-----------------------')

disp('train set: final precision result')
disp('computed as the average of weighted precision score over the 5 folds')
disp(mean(RF_weighted_precision_train))
disp('train set: final recall result')
disp('computed as the average of weighted recall score over the 5 folds')
disp(mean(RF_weighted_recall_train))
disp('train set: final f1 score result')
disp('computed as the average of weighted f1 scokre score over the 5 folds')
disp(mean(RF_weighted_fscore_train))

disp('validation set: final precision result')
disp('computed as the average of weighted precision score over the 5 folds')
disp(mean(RF_weighted_precision_test))
disp('validation set: final recall result')
disp('computed as the average of weighted recall score over the 5 folds')
disp(mean(RF_weighted_recall_test))
disp('validation set: final f1 score result')
disp('computed as the average of weighted f1 score score over the 5 folds')
disp(mean(RF_weighted_fscore_test))

% ------- error curves train and validation ------------

figure;
hold on;

plot(RF_errors_train{1},'Color','y')
plot(RF_errors_train{2},'Color','g')
plot(RF_errors_train{3},'Color','r')
plot(RF_errors_train{4},'Color','b')
plot(RF_errors_train{5},'Color','k')
ylabel('misclassification probability')

plot(RF_errors_test{1}, '--', 'Color','y')
plot(RF_errors_test{2}, '--', 'Color','g')
plot(RF_errors_test{3}, '--', 'Color','r')
plot(RF_errors_test{4}, '--', 'Color','b')
plot(RF_errors_test{5}, '--', 'Color','k')
xlabel('Number of Grown Trees')
ylabel('misclassification probability')
legend({'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'})
title(' Random Forest - errors curve per fold')

hold off;

% ----------- Feature Importance --------------
figure;
hold on;
cellfun(@plot, RF_oob_permuted_predictor_delta_error);
title('Random Forest - Feature Importance')
xlabel('Predictor variable')
ylabel('Out-of-Bag Feature Importance')
h = gca;
h.XTickLabel = clf_rf.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
legend({'fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'}, 'Location','north')
grid on

% ------------- Precision, recall and Fscore ---------------
% for each round and per class
figure;
subplot(3,1,1);
b1 = bar(1:6, RF_precision_train', 0.5, 'FaceColor',[0.2 0.2 0.5]);
hold on;
b2 = bar(1:6, RF_precision_test', .25, 'FaceColor',[0 0.7 0.7]);
grid on
xlabel('classes')
xticks(1:6)
xticklabels({'3','4','5','6','7','8'});
ylabel('precison score')
title(' Random Forest precison scores per fold')
legend([b1(1),b2(1)],'Training','Validation','Location','northwest');
hold off;

subplot(3,1,2);
bar(1:6, RF_recall_train', 0.5, 'FaceColor',[0.2 0.2 0.5])
hold on;
bar(1:6, RF_recall_test', .25, 'FaceColor',[0 0.7 0.7])
xlabel('classes')
xticks(1:6)
xticklabels({'3','4','5','6','7','8'});
ylabel('recall score')
title(' Random Forest recall scores per fold')
hold off;

subplot(3,1,3);
bar(1:6, RF_fscore_train', 0.5, 'FaceColor',[0.2 0.2 0.5])
hold on;
bar(1:6, RF_fscore_test', .25, 'FaceColor',[0 0.7 0.7])
xlabel('classes')
xticks(1:6)
xticklabels({'3','4','5','6','7','8'});
ylabel('f1 score')
title(' Random Forest F1 scores per fold')
hold off;

%% ------- RANDOM FOREST - FINAL TEST -----
% here we compute a final test on our dataset
% we initially decided to split in 5 folds the whole dataset
% so the final test consists in splitting randomly once 80% and 20% holdout

%we shuffle the dataset before cv partitioning
cleaned_wine = cleaned_wine(randperm(size(cleaned_wine, 1)), :); 

%we split once the dataset and keep 20% for final test
cv_split_2 = cvpartition(cleaned_wine.quality, 'Holdout',0.2, 'Stratify',true); 
final_training = cleaned_wine(cv_split_2.training,:);
final_test = cleaned_wine(cv_split_2.test,:);

% we normalize test and training based on training
final_training_scaled = (final_training{:,1:end-1} - mean(final_training{:,1:end-1})) ./ std(final_training{:,1:end-1});
final_test_scaled =  (final_test{:,1:end-1} - mean(final_training{:,1:end-1})) ./ std(final_training{:,1:end-1});

% store it in a table
final_training_scaled = array2table([final_training_scaled, final_training.quality],'VariableNames',final_training.Properties.VariableNames);
final_test_scaled = array2table([final_test_scaled, final_test.quality],'VariableNames',final_test.Properties.VariableNames);
  
% save the fold in a .txt file in order to apply borderlineSMOTE trhough
% python exceptionnaly 
% uncomment the following two lines if needed 
%writetable(final_training_scaled, strcat('./data_final_test/scaled_training.txt'), 'Delimiter',' ');
%writetable(final_test_scaled, strcat('./data_final_test/scaled_test.txt'), 'Delimiter',' ');

% read new training and test tables 
clear final_test_scaled; clear final_training_scaled;
final_training_scaled_oversampled = readtable('./data_final_test/oversampled_scaled_training.txt', 'Delimiter',' ');
final_test_scaled = readtable('./data_final_test/scaled_test.txt', 'Delimiter',' ');

% train random forest model based on the best hyperparameters found in the
% cross validation
rf = TreeBagger(100, ...
        final_training_scaled_oversampled, 'quality', ...
        'Method','classification',...
        'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on', ...
        'Prior', 'Empirical', ...
        'MinLeafSize', 1, ...
        'MaxNumSplits', 2000);

% predict 20% test
predict_ = rf.predict(final_test_scaled);

% compute confusion matrix ans scores on final test
[RF_precision_final(1,:), RF_recall_final(1,:), RF_fscore_final(1,:)] = compute_confusion_matrix(cellstr(string(final_test_scaled.quality)), predict_, 'Random Forest confusion matrix - Final test', 1, 'on');
support_test = arrayfun(@(x) sum(final_test_scaled.quality==x), unique(final_test_scaled.quality));

% compute weighted scores on final test
RF_weighted_precision_final = RF_precision_final(1,:)*support_test/sum(support_test);
RF_weighted_recall_final = RF_recall_final(1,:)*support_test/sum(support_test);
RF_weighted_fscore_final = RF_fscore_final(1,:)*support_test/sum(support_test);

% display final test results
disp('------------ scores below are from Random Forest on the final test set -------------')
disp('final weighted precision result')
disp(RF_weighted_precision_final)
disp('final weighted recall result')
disp(RF_weighted_recall_final)
disp('final weighted f1 score result')
disp(RF_weighted_fscore_final)


%% -----------------------------------------------------
% MODEL 2 : MULTINOMIAL REGRESSION FOR ORDINAL RESPONSES
% -------------------------------------------------------

% empty variables to store results
OR_precision_train = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));
OR_recall_train = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));
OR_fscore_train = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));

OR_precision_test = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));
OR_recall_test = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));
OR_fscore_test = zeros(cv_split.NumTestSets, size(unique(cleaned_wine.quality),1));

OR_weighted_precision_train = zeros(cv_split.NumTestSets, 1);
OR_weighted_recall_train = zeros(cv_split.NumTestSets, 1);
OR_weighted_fscore_train = zeros(cv_split.NumTestSets, 1);
OR_weighted_precision_test = zeros(cv_split.NumTestSets, 1);
OR_weighted_recall_test = zeros(cv_split.NumTestSets, 1);
OR_weighted_fscore_test = zeros(cv_split.NumTestSets, 1);

% we loop over the 5 folds and fit ordinal regression
for i = 1:cv_split.NumTestSets
    training_set = readtable(strcat(['./oversampled_data/over_train_',int2str(i),'.txt']), 'Delimiter',' ');
    test_set = readtable(strcat(['./scaled_testing_folds/scaledFold_test_',int2str(i),'.txt']), 'Delimiter',' ');

    % ------- rescaling ordinal vector starting from 1-------
    
    % our qualities in the dataset starts from 3 to 8
    ordinal_qualities = 1.*(training_set.quality==3) + 2*(training_set.quality==4) + 3*(training_set.quality==5) + 4*(training_set.quality==6) + 5*(training_set.quality==7) + 6*(training_set.quality==8);  

    % ------- model fitting on training -------

    %beta_coeff cintains first the 5 intercepts accross categories and then slope coefficients for the 7 features 
    [beta_coeff, dev, stats] = mnrfit(training_set{:,1:end-1}, ordinal_qualities,...
                                    'model','ordinal',...
                                    'interactions','off', ...
                                    'Link', 'logit');

    % ---------------- storing coefficients outputs and their statistics ----------
    
    % 5 intercepts across categories
    % 7 common coefficients of the slope from features (proportional odds assumption)
    % standard errors of each coefficients
    % p-value of each coefficients 
    % proportional_odd_ratio = exp(coefficient) since we use a logit link
    ordinal_reg_results = array2table([beta_coeff, stats.se, stats.p, exp(beta_coeff)] , ...
                            'VariableNames', {'coefficients', 'standard_error', 'pvalue', 'proportional_odd_ratio'}, ...
                            'RowNames',{'qual3|qual4','qual4|qual5','qual5|qual6','qual6|qual7','qual7|qual8','fixed_acidity','volatile_acidity','residual_sugar', 'chlorides', 'total_sulfur_di', 'sulphates', 'alcohol'});
    disp(ordinal_reg_results)
    
    % --------------- Residual deviance ------------
    %Residual Deviance = 2*log-likelihood
    disp('Residual Deviance of the model is :'); disp(dev);
    
    % ---------------- computing probabilities ---------------------
    % prob_qualities gives the probability that an observation with 7 features belongs to quality j
    % logit[P(Y<=j)] = intercept_j - Sum(beta_k . X_k)
    
    [prob_qualities_train, ~, ~] = mnrval(beta_coeff, training_set{:,1:end-1}, stats,...
                            'model','ordinal', ...
                            'Link', 'logit');
    [prob_qualities_test, ~, ~] = mnrval(beta_coeff, test_set{:,1:end-1}, stats,...
                            'model','ordinal', ...
                            'Link', 'logit');
    
    % ---------------- computing predictions ---------------------
    %predicting qualities = argmax(P(Y<=j))
    
    [~, predict_qualities_train] = max(prob_qualities_train, [], 2);
    [~, predict_qualities_test] = max(prob_qualities_test, [], 2);
    % we rescale qualities from 3 to 8
    predict_qualities_train = predict_qualities_train + 2;
    predict_qualities_test = predict_qualities_test + 2;
    
    % ----------- confusion matrix and scores ----------
 
    confusion_matrix_visible = 'off'; %change to 'on' if needed
    [OR_precision_train(i,:), OR_recall_train(i,:), OR_fscore_train(i,:)] = compute_confusion_matrix(training_set.quality, predict_qualities_train, strcat({'round '},{int2str(i)},{' OR : confusion matrix on train data'}), 1, confusion_matrix_visible);
    [OR_precision_test(i,:), OR_recall_test(i,:), OR_fscore_test(i,:)] = compute_confusion_matrix(test_set.quality, predict_qualities_test, strcat({'round '},{int2str(i)},{' OR : confusion matrix on test data'}), 1, confusion_matrix_visible);

    % ----- weighted scores ------
    
    support_train = arrayfun(@(x) sum(training_set.quality==x), unique(training_set.quality));
    support_test = arrayfun(@(x) sum(test_set.quality==x), unique(test_set.quality));
    
    OR_weighted_precision_train(i) = OR_precision_train(i,:)*support_train/sum(support_train);
    OR_weighted_recall_train(i) = OR_recall_train(i,:)*support_train/sum(support_train);
    OR_weighted_fscore_train(i) = OR_fscore_train(i,:)*support_train/sum(support_train);
    OR_weighted_precision_test(i) = OR_precision_test(i,:)*support_test/sum(support_test);
    OR_weighted_recall_test(i) = OR_recall_test(i,:)*support_test/sum(support_test);
    OR_weighted_fscore_test(i) = OR_fscore_test(i,:)*support_test/sum(support_test);
end 


%% -------- results  of Ordinal regression --------

disp('------ RESULTS BELOW ARE FROM ORDINAL REGRESSION')

disp('train set: final precision result')
disp('computed as the average of weighted precision score over the 5 folds')
disp(mean(OR_weighted_precision_train))
disp('train set: final recall result')
disp('computed as the average of weighted recall score over the 5 folds')
disp(mean(OR_weighted_recall_train))
disp('train set: final f1 score result')
disp('computed as the average of weighted f1 scokre score over the 5 folds')
disp(mean(OR_weighted_fscore_train))

disp('validation set: final precision result')
disp('computed as the average of weighted precision score over the 5 folds')
disp(mean(OR_weighted_precision_test))
disp('validation set: final recall result')
disp('computed as the average of weighted recall score over the 5 folds')
disp(mean(OR_weighted_recall_test))
disp('validation set: final f1 score result')
disp('computed as the average of weighted f1 score score over the 5 folds')
disp(mean(OR_weighted_fscore_test))


% ------------- Precision, recall and Fscore ---------------
% for each round and per class
figure;
subplot(3,1,1);
b1 = bar(1:6, OR_precision_train', 0.5, 'FaceColor',[0.2 0.2 0.5]);
hold on;
b2 = bar(1:6, OR_precision_test', .25, 'FaceColor',[0 0.7 0.7]);
grid on
xlabel('classes')
xticks(1:6)
xticklabels({'3','4','5','6','7','8'});
ylabel('precison score')
title(' Ordinal regression precison scores per fold')
legend([b1(1),b2(1)],'Training','Validation','Location','northwest');
hold off;

subplot(3,1,2);
bar(1:6, OR_recall_train', 0.5, 'FaceColor',[0.2 0.2 0.5])
hold on;
bar(1:6, OR_recall_test', .25, 'FaceColor',[0 0.7 0.7])
xlabel('classes')
xticks(1:6)
xticklabels({'3','4','5','6','7','8'});
ylabel('recall score')
title(' Ordinal regression recall scores per fold')
hold off;

subplot(3,1,3);
bar(1:6, OR_fscore_train', 0.5, 'FaceColor',[0.2 0.2 0.5])
hold on;
bar(1:6, OR_fscore_test', .25, 'FaceColor',[0 0.7 0.7])
xlabel('classes')
xticks(1:6)
xticklabels({'3','4','5','6','7','8'});
ylabel('f1 score')
title(' Ordinal regression F1 scores per fold')
hold off;

%% ------- ORDINAL REGRESSION - FINAL TEST -----

% we load the same sets than in the final test of random forest
final_training_scaled_oversampled = readtable('./data_final_test/oversampled_scaled_training.txt', 'Delimiter',' ');
final_test_scaled = readtable('./data_final_test/scaled_test.txt', 'Delimiter',' ');

%create ordinal vector for mnrfit model (requiring starting from 1)
ordinal_qualities_final = 1.*(final_training_scaled_oversampled.quality==3) + 2*(final_training_scaled_oversampled.quality==4) + 3*(final_training_scaled_oversampled.quality==5) + 4*(final_training_scaled_oversampled.quality==6) + 5*(final_training_scaled_oversampled.quality==7) + 6*(final_training_scaled_oversampled.quality==8);  

% we fit the model on the training set 
%beta_coeff cintains first the 5 intercepts accross categories and then slope coefficients for the 7 features 
[beta_coeff_final, dev_final, stats_final] = mnrfit(final_training_scaled_oversampled{:,1:end-1}, ordinal_qualities_final,...
                                    'model','ordinal',...
                                    'interactions','off', ...
                                    'Link', 'logit');

ordinal_reg_results_final = array2table([beta_coeff_final, stats_final.se, stats_final.p, exp(beta_coeff_final)] , ...
                            'VariableNames', {'final_coefficients', 'standard_error', 'pvalue', 'proportional_odd_ratio'}, ...
                            'RowNames',{'cat3|cat4','cat4|cat5','cat5|cat6','cat6|cat7','cat7|cat8','fixed_acidity','volatile_acidity','residual_sugar', 'chlorides', 'total_sulfur_di', 'sulphates', 'alcohol'});

disp('final test : coefficients from ordinal regression')
disp(ordinal_reg_results_final)
disp('Residual Deviance of the model is :'); disp(dev_final)

% computing probabilities 
[prob_qualities_test_final] = mnrval(beta_coeff_final, final_test_scaled{:,1:end-1}, ...
                            'model','ordinal', ...
                            'Link', 'logit');
                        
%predicting qualities based on the max of the probabilities of each category
[~, predict_qualities_test_final] = max(prob_qualities_test_final, [], 2);

% we rescale qualities from 3 to 8
predict_qualities_test_final = predict_qualities_test_final + 2;

% scores and confusion matrix
[OR_precision_test_final(1,:), OR_recall_test_final(1,:), OR_fscore_test_final(1,:)] = compute_confusion_matrix(final_test_scaled.quality, predict_qualities_test_final, 'Ordinal regression confusion matrix - Final test', 1, 'on');

% weighted scores
support_test = arrayfun(@(x) sum(final_test_scaled.quality==x), unique(final_test_scaled.quality));
OR_weighted_precision_final = OR_precision_test_final(1,:)*support_test/sum(support_test);
OR_weighted_recall_final = OR_recall_test_final(1,:)*support_test/sum(support_test);
OR_weighted_fscore_final = OR_fscore_test_final(1,:)*support_test/sum(support_test);

% display final test results
disp('------------ scores below are from Ordinal Regression on the final test set -------------')
disp('Ordinal regression - test set: final precision result')
disp(OR_weighted_precision_final)
disp('Ordinal regression - test set: final recall result')
disp(OR_weighted_recall_final)
disp('Ordinal regression - test set: final f1 score result')
disp(OR_weighted_fscore_final)


%% --- TABLE OF COMPARISON ---

RESULTS = [mean(RF_weighted_recall_train) , mean(RF_weighted_precision_train) , mean(RF_weighted_fscore_train) ; ...
    mean(RF_weighted_recall_test), mean(RF_weighted_precision_test), mean(RF_weighted_fscore_test) ; ...
    RF_weighted_recall_final, RF_weighted_precision_final, RF_weighted_fscore_final; ...
    mean(OR_weighted_recall_train) , mean(OR_weighted_precision_train) , mean(OR_weighted_fscore_train) ; ...
    mean(OR_weighted_recall_test), mean(OR_weighted_recall_test), mean(OR_weighted_recall_test) ; ...
    OR_weighted_recall_final, OR_weighted_precision_final, OR_weighted_fscore_final];

scores_across_models = array2table( RESULTS, ...
                            'VariableNames', {'weighted_recall', 'weighted_precison', 'weighted_f1_score'}, ...
                            'RowNames',{'RF_TRAIN','RF_VALIDATION','RF_TEST','OR_TRAIN','OR_VALIDATION','OR_TEST'});


disp('-----------------------------------------------------------')
disp('--------------COMPARISON SCORES ACROSS MODELS--------------')
disp('-----------------------------------------------------------')
disp(scores_across_models)


