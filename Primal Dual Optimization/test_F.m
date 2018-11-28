function [res1, res2, res3] = test_F(y, sigma, conjugate_flag)

res3 = 0;

if ~conjugate_flag

    res1 = 0.5 * sum(y .^ 2);
    res2 = 1 / (1 + sigma) * y;
    
else
    
    res1 = 0.5 * sum(y .^ 2);
    res2 = 1 / (1 + sigma) * y;
    
end

end