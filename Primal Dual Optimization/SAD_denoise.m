function [res1, res2, res3] = SAD_denoise(u, g, tau, conjugate_flag)
% IN:
%       u               ~ m*n x 1       template image (for denoising)
%       g               ~ m*n x 1       reference image (noisy)
%       tau             ~ 1 x 1         step length for prox operator
%       conjugate_flag  ~ logical       evaluate SAD or SAD* at u?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1         function value SAD(u)
%       res2            ~ m*n x 1       prox-step of SAD for u
%       res3            ~ 1 x 1         measure for hurt constraints
%   IF conjugate_flag:
%       res1            ~ 1 x 1         convex conjugate SAD*(u)
%       res2            ~ m*n x 1       prox-step of SAD* for u
%       res3            ~ 1 x 1         measure for hurt constraints

% by default: evaluate SAD instead of its conjugate
if nargin < 4, conjugate_flag = false; end

% initialize constraint measure with 0
res3 = 0;

if ~conjugate_flag
    % EITHER ~> evaluate SAD and Prox_[SAD] at u
    
    % compute SAD
    res1 = sum(abs(u - g));
    
    if nargout == 2
        
        % prox-step on u for G = SAD ~> pointwise shrinkage
        res2 = zeros(size(u));
        diff_ug = u - g;
        idx1 = (diff_ug > tau);
        idx2 = (diff_ug < (-1) * tau);
        idx3 = ~(idx1 | idx2);
        res2(idx1) = u(idx1) - tau;
        res2(idx2) = u(idx2) + tau;
        res2(idx3) = g(idx3);
        
    else
        res2 = [];
    end
    
else
    % OR ~> evaluate SAD* and Prox_[SAD*] at u
    
    % compute SAD*(u) = delta_{||.||_inf <= 1}(u) + <u,g>
    if max(abs(u)) > 1
        res3 = max(abs(u)) - 1;
    end
    res1 = u' * g;
    
    if nargout == 2
        
        % compute prox-step for G* = SAD* with Moreau's identity
        %   [(id + tau * dG*)^(-1)](u) = 
        %       u - tau * [(id + (1 / tau) * dG)^(-1)](u / tau)
        [~, prox] = SAD_denoise(u / tau, g, 1 / tau, false);
        res2 = u - tau * prox;
        
    else
        res2 = [];
    end
    
end

end