function [res1, res2] = test_F(y, sigma, conjugate_flag)

if ~conjugate_flag

    res1 = 0.5 * sum(y .^ 2);
    res2 = 1 / (1 + sigma) * y;
    
else
    
    res1 = 0.5 * sum(y .^ 2);
    res2 = 1 / (1 + sigma) * y;
    
end

end