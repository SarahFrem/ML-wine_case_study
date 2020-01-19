function [precision, recall, fscore] = compute_confusion_matrix(true_target, predicted_target, title, beta, isVisible)
% this function computes the confusion matrix the precision, recall and Fscore
% arguments : 
% true_target : array; the true target 
% predicted_target: array; the predicted target from the model (same type as true_target)
% title : string; title of the confusion matrix
% beta : integer; beta variable for the Fscore 
% isVisible : 'on' or 'off'; 'on' for displaying the confusion matrix 

    figure('visible',isVisible);
    cm = confusionchart(true_target, predicted_target);
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    cm.Title = title;

    precision = zeros(1,size(cm.NormalizedValues,1));
    recall = zeros(1,size(cm.NormalizedValues,1));
    fscore = zeros(1,size(cm.NormalizedValues,1));
    
    for i=1:size(cm.NormalizedValues,1)
        tp = cm.NormalizedValues(i,i);
        fn = sum(cm.NormalizedValues(i,:)) - tp;
        fp = sum(cm.NormalizedValues(:,i)) - tp;
        
        if tp+fp==0
            precision(i) = 0;
        else
            precision(i) = tp/(tp+fp);
        end
        
        if tp+fn==0
            recall(i) = 0;
        else
            recall(i) = tp/(tp+fn);
        end 
    
        if (precision(i)==0 && recall(i)==0)
            fscore(i) = 0;
        else
            fscore(i) = (beta^2 + 1)*precision(i)*recall(i) / (beta^2 * precision(i) + recall(i));
        end 
    end
    
  end

