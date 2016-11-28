function [accuracy,classification_error] = computeAccuracySoftmax(class_probs,data_labels)

[~,predicted_labels] = max(class_probs) ;

%subtracting 1 because matlab starts indexing from 1
predicted_labels = predicted_labels - 1 ;

accuracy = mean(predicted_labels == data_labels') * 100;

classification_error = mean(predicted_labels ~= data_labels') * 100;
end