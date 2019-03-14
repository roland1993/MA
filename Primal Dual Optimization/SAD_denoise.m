function [res1, res2, res3] = ...
    SAD_denoise(u, g, lambda, tau, conjugate_flag)
%--------------------------------------------------------------------------
% This file is part of my master's thesis entitled
%           'Low rank- and sparsity-based image registration'
% For the whole project see
%           https://github.com/roland1993/MA
% If you have questions contact me at
%           roland.haase [at] student.uni-luebeck [dot] de
% Source code is provided under the
%           MIT Open Source License
%--------------------------------------------------------------------------
% IN:
%       u               ~ m*n x 1       template image (for denoising)
%       g               ~ m*n x 1       reference image (noisy)
%       lambda          ~ 1 x 1         weighting factor
%       tau             ~ 1 x 1         step length for prox operator
%       conjugate_flag  ~ logical       evaluate SAD or SAD* at u?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1         function value SAD(u)
%       res2            ~ 1 x 1         constraint violation measure
%       res3            ~ m*n x 1       prox-step of SAD for u
%   IF conjugate_flag:
%       res1            ~ 1 x 1         convex conjugate SAD*(u)
%       res2            ~ 1 x 1         constraint violation measure
%       res3            ~ m*n x 1       prox-step of SAD* for u
%--------------------------------------------------------------------------

% by default: evaluate SAD instead of its conjugate
if nargin < 5, conjugate_flag = false; end

% initialize constraint measure with 0
res2 = 0;

if ~conjugate_flag
    % EITHER ~> evaluate SAD and Prox_[SAD] at u, weighted by lambda
    
    % compute SAD
    res1 = lambda * sum(abs(u - g));
    
    if nargout == 3
        
        % prox-step on u for G = SAD ~> pointwise shrinkage
        res3 = zeros(size(u));
        diff_ug = u - g;
        idx1 = (diff_ug > lambda * tau);
        idx2 = (diff_ug < (-1) * lambda * tau);
        idx3 = ~(idx1 | idx2);
        res3(idx1) = u(idx1) - lambda * tau;
        res3(idx2) = u(idx2) + lambda * tau;
        res3(idx3) = g(idx3);
        
    end
    
else
    % OR ~> evaluate SAD* and Prox_[SAD*] at u
    
    % scaling lambda ~> [lambda * G]*(u) = lambda * G*(u/lambda)
    u = u / lambda;
    
    % compute SAD*(u) = delta_{||.||_inf <= 1}(u) + <u,g>
    if max(abs(u)) > 1
        res2 = max(abs(u)) - 1;
    end
    res1 = lambda * u' * g;
    
    if nargout == 3
        
        % compute prox-step for G* = SAD* with Moreau's identity
        %   [(id + tau * dG*)^(-1)](u) = 
        %       u - tau * [(id + (1 / tau) * dG)^(-1)](u / tau)
        [~, ~, prox] = SAD_denoise(u / (lambda * tau), g, lambda, ...
            1 / (lambda * tau), false);
        res3 = u - lambda * tau * prox;
        
    end
    
end

end