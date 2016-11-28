function [x_t,h_t,x_tild,h_tild,reco] = CD(W_b,W_c,x_1,K,reco)

for k = 1 : K

                p_h = sigmoid(W_b*x_1) ;
                u1_mat = rand(size(p_h)) ;
                h = double(p_h> u1_mat);
            
                if (k == 1)
                    x_t = x_1(1:size(x_1,1)-1,:);
                    h_t = p_h;
                end

                h_1 = [h; ones([1 size(h,2)])];
                p_x = sigmoid(W_c' * h_1);

%                 u2_mat = rand(size(p_x));
%                 x = double(p_x>u2_mat);
% % not sampling on the x actually improved reconstruction
% %                 
                x = p_x;
                x_1 = [x; ones([1 size(x,2)])];

                if (k == K)

                    x_tild = x;%x_1(1:size(x,1),:);
                    reco=[reco x_tild];
                    p_h_tild = sigmoid(W_b*x_1 ) ;

        %           u3_mat = rand(size(p_h_tild));
        %           u3_mat = repmat(rand,[size(p_h_tild)]);
        % 
        %           h_tild = double(p_h_tild> u3_mat);
                    h_tild = p_h_tild;
                end
           end
end