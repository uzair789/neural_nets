function [image_weights]= visualize(W )
W_append=[];

[hidden_size input_size] = size(W);
images_per_row = ceil(sqrt(hidden_size));
images_per_col = ceil(hidden_size/images_per_row);
img = 1;
append_rows = (images_per_row * images_per_col) - hidden_size;
W_append = [W ; zeros([append_rows input_size])];
container = {};
for i = 1: images_per_col
    for j = 1 : images_per_row
      container{i,j} = reshape(W_append(img, : ),[28 28]);
      img = img+1;
      
    end
end

image_weights = cell2mat(container);
image_weights = image_weights - min(min(image_weights));

end

