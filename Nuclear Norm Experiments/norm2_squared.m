function [res1, res2, res3] = norm2_squared(x, tau, conjugate_flag)


if ~conjugate_flag
    
    res1 = sum(x .^ 2);
    res2 = 0;
    res3 = x / (1 + 2 * tau);
    
else
    
    res1 = sum(x .^ 2) / 4;
    res2 = 0;
    [~, ~, prox] = norm2_squared(x / tau, 1 / tau, false);
    res3 = x - tau * prox;
    
end

end