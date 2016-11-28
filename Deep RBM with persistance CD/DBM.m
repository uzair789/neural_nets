clear all;close all;

folder_path = ['C:\Users\uzair\Desktop\DBM_Images'];
mkdir(folder_path);
plot_count = 105;

disp('Loading Data...');
[train,val,test] = loadData1();
disp('Data Loaded!!!');

V = train.images;
X_CV = val.images;

%Converting the images to binary images
V = double(V >0.001);
X_CV = double(X_CV >0.001);

%hidden layers
hidden_layer1 = 100;
hidden_layer2 = 100;
input_layer = size(V,1);

%100 persistent chains
no_chains = 100; %number of persistent chains
v_chains = rand(input_layer,no_chains);
h2_chains = rand(hidden_layer2,no_chains);
v_chains_cv = X_CV;
% v_chains_cv = rand(input_layer,no_chains);
h2_chains_cv = rand(hidden_layer2,size(v_chains_cv,2));

%weight initializations
W1 = sqrt(6/(hidden_layer1 + input_layer))*randn([hidden_layer1 input_layer]);
W2 = sqrt(6/(hidden_layer1 + input_layer))*randn([hidden_layer2 hidden_layer1]);
b1 = zeros([hidden_layer1 1]);
b2 = zeros([hidden_layer2 1]);
c = zeros([input_layer 1]);

%hyperparameters
epochs = 5;
batch_size = 1;
CD_steps = 10;
lr = 0.01;
count = 1;
figure(1);
% live_plotting = 'yes'; % 'yes' or 'no'
live_plotting = 'no'; % 'yes' or 'no'

hold on;
gibbs_sampling = 1;
for epoch = 1 : epochs
    epoch
    V = shuffle(V);
    X_CV = shuffle(X_CV);    
    
    for i = 1:batch_size:size(V,2)
   
       v = V(:,i:i+batch_size-1);
       u1 = randn([hidden_layer1 batch_size]);
       u2 = randn([hidden_layer2 batch_size]);
           
           for k = 1 : CD_steps
%                u1 = sigmoid( [W1 b1] * [v;ones([1 batch_size])] + [W2' b1] * [u2;ones([1 batch_size])] );
               
               u1 = sigmoid( [W1 b1] * [v;ones([1 batch_size])] + [W2'] * [u2] );
               u2 = sigmoid( [W2 b2] * [u1;ones([1 batch_size])]);
     
           end%cd steps
       
          [h1_initial,h1_final,h2_initial,h2_final,v_initial,v_final,v_chains,h2_chains] = PD(v_chains,h2_chains,W1,W2,b1,b2,c,no_chains,gibbs_sampling);
          error_train = cross_entropy_loss(v_initial,v_final) / no_chains; %changed v_initial to v , h2_chains1 -> h2_chains
          error_train_array(count) = error_train;
          
          %validation
          [~,~,~,~,v_initial_cv,v_final_cv,v_chains_cv,h2_chains_cv] = PD(v_chains_cv,h2_chains_cv,W1,W2,b1,b2,c,no_chains,gibbs_sampling);
          error_cv = cross_entropy_loss(v_initial_cv,v_final_cv) / size(v_chains_cv,2);
          error_cv_array(count) = error_cv;
          
          if strcmp(live_plotting,'yes')
%                error = mse(v_initial , v_final)
               plot(count,error_train,'--b.','markersize',10);
%                weights = visualize(v_final');
%                imshow(weights');
               hold on;
               plot(count,error_cv,'--r.','markersize',10);
               pause(0.000001)

           end

        count = count+1;

        %parameter update
        W1 = W1 + lr * ( 1/batch_size * (u1 * v') - 1/no_chains * (h1_final * v_final')  );
        W2 = W2 + lr * ( 1/batch_size * (u2 * u1') - 1/no_chains * (h2_final * h1_final') );
      
        b1 = b1 + lr * (u1 - 1/(no_chains)*sum(h1_final,2));
        b2 = b2 + lr * (u2 - 1/(no_chains)*sum(h2_final,2));
        c  = c + lr *  (v  - 1/(no_chains)*sum(v_final,2));
%                 
%       disp(['epoch = ' num2str(epoch) ' | batch = ' num2str(i)]);
    end %mini batch (i)
end %epoch
hyperparams = sprintf('lr = %.3f | CD steps = %d | hidden layer1 = %d | hidden layer2 = %d | gibbs sampling = %d',lr,CD_steps,hidden_layer1,hidden_layer2,gibbs_sampling);

figure(2);
plot(1:length(error_train_array),error_train_array,'-b',1:length(error_cv_array),error_cv_array,'-r')
title('Cross entropy loss')
xlabel('iterations')
ylabel('cross entropy error')
legend('training error','validation error')
grid on;
ylim = get(gca,'Ylim');
xlim = get(gca,'Xlim');
units = (ylim(2)-ylim(1))/10;
xpos = 100;
ypos = max(error_train_array)-units;
text(xpos,ypos,hyperparams);
filename = [num2str(plot_count) 'REhidden1=' num2str(hidden_layer1) 'hidden2=' num2str(hidden_layer2)];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

figure(3);
image_weights = visualize(W1);
% image_weights = image_weights - min(min(image_weights));
imshow(image_weights);
title('Weights for layer 1')
xlabel(hyperparams);
filename = [num2str(plot_count) 'WeightsW1hidden1=' num2str(hidden_layer1) 'hidden2=' num2str(hidden_layer2) ];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

figure(4);
image_weights = visualize(v_final');
% image_weights = image_weights - min(min(image_weights));
imshow(image_weights');
title(['100 Sampled chains '])
xlabel(hyperparams);
filename = [num2str(plot_count) 'SamplesW1hidden1=' num2str(hidden_layer1) 'hidden2=' num2str(hidden_layer2) ];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

