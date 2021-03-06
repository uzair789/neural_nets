
function [train,val,test] = loadData2()

[train_data_with_labels] = textread('../data/digitstrain.txt','','delimiter',',');
[val_data_with_labels] = textread('../data/digitsvalid.txt','','delimiter',',');
[test_data_with_labels] = textread('../data/digitstest.txt','','delimiter',',');


train_data = train_data_with_labels(:,1:784);
train_labels = train_data_with_labels(:,785);

val_data = val_data_with_labels(:,1:784);
val_labels = val_data_with_labels(:,785);

test_data = test_data_with_labels(:,1:784);
test_labels = test_data_with_labels(:,785);


train = struct('images',train_data','labels',train_labels);
val = struct('images',val_data','labels',val_labels);
test = struct('images',test_data','labels',test_labels);

end