function [data_wine_cleaned, var_names] = preprocess_data(data_wine)
% this function calls the preprocessing step : removing useless features
% and outliers
% argument:
% data_wine : table of initial data

    disp('launch preprocessing data')
    disp('please refer to coursework_preprocessing.m for explanations and visualizarions supporting our choices')
    
    disp('------------------------------------')
    
    disp('remove useless features based on density, trends and domain knowledge')
    data_wine = removevars(data_wine,{'free_sulfur_dioxide', 'density', 'pH', 'citric_acid'});
    var_names = {'fixed_acidity', 'volatile_acidity', 'residual_sugar', 'chlorides', 'total_sulfur_dioxide', 'sulphates', 'alcohol', 'quality'};

    disp('remove outliers based on Mahalanobis distance per category')
    obj = fitcdiscr(data_wine{:,1:end - 1}, data_wine{:,end});
    mahalanobis_distances = mahal(obj, data_wine{:,1:end - 1});
    mahalanobis_distances_t = array2table([mahalanobis_distances, data_wine{:,end}],'VariableNames', {'cat3','cat4','cat5','cat6','cat7','cat8', 'quality_class'}); 
    outliers_detection_cat5 = (mahalanobis_distances_t{:,3}>=40 & mahalanobis_distances_t{:,7}==5);
    outliers_detection_cat6 = (mahalanobis_distances_t{:,4}>=30 & mahalanobis_distances_t{:,7}==6);
    outliers_detection_cat7 = (mahalanobis_distances_t{:,5}>=30 & mahalanobis_distances_t{:,7}==7); 
    outliers_detection = outliers_detection_cat5 | outliers_detection_cat6 | outliers_detection_cat7;
    data_wine_cleaned = data_wine(~outliers_detection,:);
end

