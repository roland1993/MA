function [res1, res2, res3] = test_G(x, g, lambda, tau, conjugate_flag)

res3 = 0;

if ~conjugate_flag
    
    res1 = 0.5 * lambda * sum((x(:) - g(:)) .^ 2);
    res2 = 1 / (1 + (lambda * tau)) * (x + (lambda * tau) * g);
    
else
    
    res1 = lambda * (0.5 * sum((x / lambda) .^ 2) + ((x / lambda)' * g));
    res2 = [];
    
end

end