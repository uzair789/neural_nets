 clear all;close all;
 
disp('Loading Data...');
[train,val,test] = loadData1();
disp('Data Loaded!!!');

folder_path = 'C:\Users\uzair\Desktop\RBM_Images';
mkdir(folder_path);
plot_count = 208;
debug = 0;
hidden = [100];
KK = [1];
for k = 1:length(KK)
    for l = 1:length(hidden)

%== Initializations
hidden_units = hidden(l);
W = sqrt(6/(size(train.images,1)+hidden_units)) * randn([hidden_units 784]); %size(train.images,1) = 784
c = zeros(size(train.images,1),1); % bias for visible
b = zeros(hidden_units,1); % bias for hidden
X_1 = [train.images; ones([1 size(train.images,2)])];
X_CV = [val.images; ones([1 size(val.images,2)])];

%Converting the images to binary images
X_1 = double(X_1>0.01);
X_CV = double(X_CV>0.01);

K = KK(k);

lr = 0.01;
epochs = 5;
batch_size = 1;
x_1=X_1;
reco=[];

error_counter = 1;
% hold off;
if debug == 1
    figure(1);
    title(['Reconstruction Error k = ' num2str(K)]);
    xlabel('iterations')
    ylabel('cross entropy error')
    legend('training error','validation error')
    hold on;
end

error_best = 400;

for epoch = 1 : epochs
    epoch
X_1 = shuffle(X_1);
X_CV = shuffle(X_CV);
%shuffling actually helps improve the generalization results

       for i = 1:batch_size:size(train.images,2)
           
           if i == 1 %for persistent CD -- remove the if else and the statement in else for regular CD
               x_1=X_1(:,i:i+batch_size-1); 
           else
               x_1 = [x_tild;ones([1 size(x_tild,2)])];
           end
           
           W_c = [W ;c'];
           W_b = [W b];
           
           %contrastive divergence
           [x_t,h_t,x_tild,h_tild,reco] = CD(W_b,W_c,x_1,K,reco);

           %persistnece
           x_t_1 = X_1(:,i:i+batch_size-1);
           p_h = sigmoid(W_b*x_t_1) ;
           h_t = p_h;
           x_t = x_t_1(1:784,:);
           %update the weights and the biases
           W = W + lr * (h_t * x_t' - h_tild * x_tild')/batch_size;
           b = b + lr * sum( h_t - h_tild , 2)/batch_size;
           c = c + lr * sum( x_t - x_tild , 2)/batch_size;

           %checking for the reconstruction error trin set
           H_T = [h_t;ones([1 size(h_t,2)])];

           x_hat = sigmoid(W_c'*H_T);
           error_train = cross_entropy_loss(x_t,x_hat);
           error_train_norm = error_train/batch_size;
           cross_ent(error_counter) = error_train_norm;%- error * log(error);
   
           %checking error on the val set
           p_h_cv = sigmoid(W_b*X_CV);
           u4_mat = rand(size(p_h_cv)) ;
           h_cv = double(p_h_cv> u4_mat);
           h_cv_1 = [h_cv; ones([1 size(h_cv,2)])];
           X_CV_hat = sigmoid(W_c' * h_cv_1);

           error_cv = cross_entropy_loss(val.images,X_CV_hat);
           error_cv_norm = error_cv/1000;
           cross_ent_cv(error_counter) = error_cv_norm;
           
           
%            disp(sprintf('epoch = %d | i = %d | train error = %f | val error = %f ', epoch,i,error_train_norm,error_cv_norm));
         
         if debug == 1   
           plot(error_counter,error_train_norm,'--b.','markersize',10)%'.','markersize',10);
           grid on;
           hold on;
           plot(error_counter,error_cv_norm,'--r.','markersize',10)%,'markersize',10);
           pause(0.00001)
         end
         error_counter = error_counter+1;  
       end
end
hold off;

hyperparams = sprintf('lr = %.3f | k = %d | hidden units = %d | sampling in CD steps on p(x)',lr,K,hidden_units);

figure(2);
plot(1:length(cross_ent),cross_ent,'-b',1:length(cross_ent_cv),cross_ent_cv,'-r')
title(['Cross entropy loss for k = ' num2str(K)])
xlabel('iterations')
ylabel('cross entropy error')
legend('training error','validation error')
grid on;
ylim = get(gca,'Ylim');
xlim = get(gca,'Xlim');
units = (ylim(2)-ylim(1))/10;
xpos = 100;
ypos = max(cross_ent)-units;
text(xpos,ypos,hyperparams);
filename = [num2str(plot_count) 'REk=' num2str(K) 'hidden=' num2str(hidden_units)];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

figure(3);
image_weights = visualize(W);
% image_weights = image_weights - min(min(image_weights));
imshow(image_weights);
title(['Weights for k = ' num2str(K)])
xlabel(hyperparams);
filename = [num2str(plot_count) 'weightsk=' num2str(K) 'hidden=' num2str(hidden_units)];
saveas(gca, fullfile(folder_path, filename), 'jpeg');
% 

%%%shows all the reconstructions of all the samples during training.
%%% runs okay for low value of epochs but higher values will mean collecting 3000 images per epoch for all the epochs
figure(5);
reco_images = visualize(reco');
imshow(reco_images');
title(['reconst for k = ' num2str(K)])
filename = [num2str(plot_count) ' reco k = ' num2str(K) 'hidden=' num2str(hidden_units)];
saveas(gca, fullfile(folder_path, filename), 'jpeg');

figure(6);
image_weights = visualize(X_1(1:784,:)');
imshow(image_weights');
title('original images')
plot_count = plot_count + 1;
%saving the W,b,c values
mat_file = ['RBM k' num2str(K) 'hid' num2str(hidden_units) 'ep' num2str(epochs) 'lr' num2str(lr)];
% save([mat_file '.mat'],'W','b','c');
% close all;
end
end

