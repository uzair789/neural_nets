clear all;close all;

disp('Loading Data...');
[train,val,test] = loadData1();
disp('Data Loaded!!!');
plot_count = 1;
debug = 0;

%for a denoising autoencoder p != 1
% for a normal autoencoder, p = 1
P = [1,0.5];
hidden_layer = [50,100,200,500];

for ii = 1:length(P)
    for jj = 1:length(hidden_layer)
    
    
p = P(ii);
if p == 1
    enc = 'Autoencoder';
    disp('Running Autoencoder...')
    folder_path = 'C:\Users\uzair\Desktop\Autoencoder_Images';
    mkdir(folder_path);
else
    enc = 'DenoisyAutoencoder';
    disp('Running Denoisy Autoencoder...')
    folder_path = 'C:\Users\uzair\Desktop\Denoisy_Autoencoder_Images';
    mkdir(folder_path);
end




%== Initializations
hidden_units = hidden_layer(jj);
W = sqrt(6/(size(train.images,1)+hidden_units)) * randn([hidden_units 784]); %size(train.images,1) = 784
c = zeros(size(train.images,1),1); % bias for visible
b = zeros(hidden_units,1); % bias for hidden
X_1 = [train.images; ones([1 size(train.images,2)])];
X_CV_1 = [val.images; ones([1 size(val.images,2)])];

% Converting the images to binary images
% X_1 = double(X_1>0.5);
% X_CV_1 = double(X_CV_1>0.5);

lr = 0.01;
reco=[];

error_counter = 1;


if debug == 1
    figure(1);
    title([enc ' Loss for hidden layer = ' num2str(hidden_units)]);
    xlabel('iterations');
    ylabel('cross entropy error');
    hold on;
end

epochs = 2;
batch_size = 1;
num_iter = size(X_1,2);


error_best = 400;
for epoch = 1 : epochs
    
X_1 = shuffle(X_1);
X_CV_1 = shuffle(X_CV_1);
%shuffling actually helps improve the generalization results

       for i = 1:batch_size:size(train.images,2)
           W2 = W';W1=W;
           x = X_1(1:784,i:i+batch_size-1);
           
           %Denoising autoencoder || p = 1 for normal auto encoder
           mask = (rand(size(x)) < p);
           x_m = mask .* x;
           %binarize
           x_m = double(x_m > 0.01);
           
           x_m_1=[x_m;ones(1,size(x,2))]; 
           
           W_c = [W2 c];
           W_b = [W1 b];
           
           %Forward Propogation
           [x_hat,h]=FP_auto(W_b,W_c,x_m_1);
           reco = [reco x_hat];
           
           %Loss
           L = cross_entropy_loss(x,x_hat);
           cross_ent(error_counter) = L;
   
           %Backprop
           da2 = x_hat - x;
           dc = sum(da2,2);
           dW2 = da2 * h';
           
           dh = W2'*da2;
           da1 = dh .* h .* (1-h);
           db = sum(da1,2);
           dW1 = da1 * x_m';
                      
           %update the weights and the biases

           dW = dW1 + dW2';
           W = W - lr * dW;
           b = b - lr * db;
           c = c - lr * dc;

           %checking error on the val set
           mask2 = (rand(size(X_CV_1)) < p);
           X_CV_1 = mask2 .* X_CV_1;
           %binarize
           X_CV_1 = double(X_CV_1 > 0.5);
           
           [x_hat_cv] = FP_auto(W_b,W_c,X_CV_1);
           L_cv = cross_entropy_loss(val.images,x_hat_cv);
           cv_norm = L_cv/1000;
           cross_ent_cv(error_counter) = cv_norm;
           
%            if cv_norm <= error_best
%                W_best_auto{plot_count} = W;
%                b_best_auto{plot_count} = b;
%                c_best_auto{plot_count} = c;
%                error_best = cv_norm;
%                disp('Best weights updated');
%            end
           
           disp(sprintf('epoch = %d | i = %d | train error = %f | val error = %f ', epoch,i,L,cv_norm));
             
        if debug == 1
           plot(error_counter,L,'--b.','markersize',10)%'.','markersize',10);
           grid on;
           hold on;
           plot(error_counter,cv_norm,'--r.','markersize',10)%,'markersize',10);
           pause(0.0001);
        end
        error_counter = error_counter+1; 
       end
end

hold off;
hyperparams = sprintf('lr = %.3f | hidden units = %d',lr,hidden_units);

figure(2);
plot(1:length(cross_ent),cross_ent,'-b',1:length(cross_ent_cv),cross_ent_cv,'-r')
% title('training error')
% figure(2);
% plot(1:length(cross_ent_cv),cross_ent_cv,'-r')
grid on;
title([enc ' Cross entropy loss for hidden layer size = ' num2str(hidden_units)]);
xlabel('iterations')
ylabel('cross entropy error')
legend('training error','validation error')
ylim = get(gca,'Ylim');
xlim = get(gca,'Xlim');
units = (ylim(2)-ylim(1))/10;
xpos = 1000;
ypos = max(cross_ent)-units;
text(xpos,ypos,hyperparams);
filename = [num2str(plot_count) enc 'RE'];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

figure(3);
image_weights = visualize(W);
% image_weights = image_weights - min(min(image_weights));
imshow(image_weights);
title([enc ' Weights for hidden layer size = ' num2str(hidden_units)])
filename = [num2str(plot_count) enc 'weights'];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

% figure(4);
% image_weights = visualize(W_best_auto{plot_count});
% % image_weights = image_weights - min(min(image_weights));
% imshow(image_weights);
% xlabel(hyperparams)
% title([ enc ' Best Weights for hidden layer size = ' num2str(hidden_units)])
% filename = [num2str(plot_count) enc ' Best weights'];
% saveas(gca, fullfile(folder_path, filename), 'jpeg');


figure(4);
reco_images = visualize(reco');
imshow(reco_images');
title([enc ' reconst for hidden layer size = ' num2str(hidden_units)])
filename = [num2str(plot_count) 'reco '];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

figure(5);
image_weights = visualize(X_1(1:784,:)');
imshow(image_weights');
title(['original images for hidden layer size = ' num2str(hidden_units)])
plot_count = plot_count + 1;

%saving the W,b,c values
mat_file = [enc 'hid' num2str(hidden_units) 'ep' num2str(epochs) 'lr' num2str(lr)];
save([mat_file '.mat'],'W','b','c');
close all;
    end
end



% save('Auto best','W_best_auto','b_best_auto','c_best_auto');