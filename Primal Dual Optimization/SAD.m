function [res1, res2] = SAD(u, g, lambda, tau, conjugate_flag)
% IN:
%       u               ~ m*n x 1       template image (for denoising)
%       g               ~ m*n x 1       reference image (noisy)
%       lambda          ~ 1 x 1         data term weighting factor
%       tau             ~ 1 x 1         step length for prox operator
%       conjugate_flag  ~ logical       evaluate SAD or SAD* at u?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1         function value [lambda * SAD](u)
%       res2            ~ m*n x 1       prox-step of [lambda * SAD] for u
%   IF conjugate_flag:
%       res1            ~ 1 x 1         convex conjugate [lambda * SAD]*(u)
%       res2            ~ m*n x 1       prox-step of [lambda * SAD]* for u

% by default: evaluate [lambda * SAD] instead of its conjugate
if nargin < 5, conjugate_flag = false; end

if ~conjugate_flag
    % EITHER ~> evaluate [lambda * SAD] and Prox_[lambda * SAD] at u
    
    % compute lambda * SAD
    res1 = lambda * sum(abs(u - g));
    
    if nargout == 2
        
        % prox-step on u for G = [lambda * SAD] ~> pointwise shrinkage
        res2 = zeros(size(u));
        diff_ug = u - g;
        idx1 = (diff_ug > tau * lambda);
        idx2 = (diff_ug < (-1) * tau * lambda);
        idx3 = ~(idx1 | idx2);
        res2(idx1) = u(idx1) - tau * lambda;
        res2(idx2) = u(idx2) + tau * lambda;
        res2(idx3) = g(idx3);
        
    end
    
else
    % OR ~> evaluate [lambda * SAD]* and Prox_[lambda * SAD]* at u
    
    % compute [lambda * SAD]*(u) = delta_{||.||_inf <= lambda}(u) + <u,g>
    if (max(abs(u)) - lambda) > 1e-10
        res1 = inf;
    else
        res1 = u' * g;
    end
    
    if nargout == 2
        
        % compute prox-step for G* = [lambda * SAD]* with Moreau's identity
        %   [(id + tau * dG*)^(-1)](u) = 
        %       u - tau * [(id + (1 / tau) * dG)^(-1)](u / tau)
        [~, prox] = SAD(u / tau, g, lambda, tau, false);
        res2 = u - tau * prox;
        
    end
    
end

end