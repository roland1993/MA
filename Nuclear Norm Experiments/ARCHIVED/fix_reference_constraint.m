function [res1, res2, res3] = ...
    fix_reference_constraint(u, s, ref_idx, conjugate_flag)

m = s(1);
n = s(2);
k = s(3);

u = reshape(u, 2* m * n, k);

res1 = zeros(1);
res2 = zeros(1);
res3 = zeros(size(u));

for i = 1 : k
    
    if i == ref_idx
        [res1_i, res2_i, res3_i] = zero_function(u(:, i), ~conjugate_flag);
    else
        [res1_i, res2_i, res3_i] = zero_function(u(:, i), conjugate_flag);
    end
    
    res1 = res1 + res1_i;
    res2 = max(res2, res2_i);
    res3(:, i) = res3_i;
    
end

res3 = res3(:);

end