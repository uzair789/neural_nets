function [h1_initial,h1_final,h2_initial,h2_final,v_initial,v_final,v_chains,h2_chains] = PD(v_chains,h2_chains,W1,W2,b1,b2,c,no_chains,K)

for j = 1:K
%     p_h1_vh2 = sigmoid( [W1 b1] * [v_chains;ones([1 size(v_chains,2)])] + [W2' b1] * [h2_chains;ones([1 size(v_chains,2)])]);
       p_h1_vh2 = sigmoid( [W1 b1] * [v_chains;ones([1 size(v_chains,2)])] + W2' * h2_chains);
  
    g1 = rand(size(p_h1_vh2));
    h1 = double(p_h1_vh2 > g1);
                
    if j == 1
        v_initial = v_chains;
        h2_initial = h2_chains;
        h1_initial = h1;
    end
       
    p_h2_h1 = sigmoid([W2 b2] * [h1;ones([1 size(v_chains,2)])]);
    g2 = rand(size(p_h2_h1));
    h2 = double(p_h2_h1 > g2);
       
    p_v = sigmoid([W1' c]* [h1;ones([1 size(v_chains,2)])]);
%   g3 = rand(size(p_v));
%   vs = double(p_v > g3);
    vs=p_v;

    v_chains = vs;
    h2_chains = h2;
           
    if j == K
        v_final = vs;
        h2_final = h2;
%       p_h1_vh2_final = sigmoid( [W1 b1] * [v_final;ones([1 size(v_chains,2)])] + [W2' b1] * [h2_final;ones([1 size(v_chains,2)])]);
        p_h1_vh2_final = sigmoid( [W1 b1] * [v_final;ones([1 size(v_chains,2)])] + W2' * h2_final);
        
        g1_final = rand(size(p_h1_vh2_final));
        h1_final = double(p_h1_vh2_final > g1_final);
    end
end
end