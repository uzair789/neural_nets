
function [train_data,train_labels,val_data,val_labels,test_data,test_labels] = loadData()

[train_data_with_labels] = textread('data/digitstrain.txt','','delimiter',',');
[val_data_with_labels] = textread('data/digitsvalid.txt','','delimiter',',');
[test_data_with_labels] = textread('data/digitstest.txt','','delimiter',',');


train_data = train_data_with_labels(:,1:784);
train_labels = train_data_with_labels(:,785);

val_data = val_data_with_labels(:,1:784);
val_labels = val_data_with_labels(:,785);

test_data = test_data_with_labels(:,1:784);
test_labels = test_data_with_labels(:,785);


end