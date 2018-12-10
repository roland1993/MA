function [res1, res2, res3] = SAD(L, I, sigma, conjugate_flag)
% IN:
%       L               ~ m*n*numImg x 1    variable to
%       I               ~ m*n*numImg x 1    all images in one column vector
%       sigma           ~ 1 x 1             prox step size
%       conjugate_flag  ~ logical           evaluate SAD or conjugate NN*?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1             function value SAD(L)
%       res2            ~ m*n x 1           prox-step of SAD for L
%       res3            ~ 1 x 1             measure for hurt constraints
%   IF conjugate_flag:
%       res1            ~ 1 x 1             convex conjugate SAD*(L)
%       res2            ~ m*n x 1           prox-step of SAD* for L
%       res3            ~ 1 x 1             measure for hurt constraints

% by default: evaluate SAD instead of its conjugate
if nargin < 4, conjugate_flag = false; end

% initialize constraint measure with 0
res3 = 0;

if ~conjugate_flag
    % EITHER ~> evaluate SAD and Prox_[SAD] at L
    
    % compute SAD
    res1 = sum(abs(L - I));
    
    if nargout == 2
        
        % prox-step for ||L - I||_1 =: SAD ~> pointwise shrinkage
        res2 = zeros(size(L));
        diff_LI = L - I;
        idx1 = (diff_LI > sigma);
        idx2 = (diff_LI < (-1) * sigma);
        idx3 = ~(idx1 | idx2);
        res2(idx1) = L(idx1) - sigma;
        res2(idx2) = L(idx2) + sigma;
        res2(idx3) = I(idx3);
        
    else
        res2 = [];
    end
    
else
    % OR ~> evaluate SAD* and Prox_[SAD*] at L
    
    % compute SAD*(L) = delta_{||.||_inf <= 1}(L) + <L,I>
    if max(abs(L)) > 1
        res3 = max(abs(L)) - 1;
    end
    res1 = L' * I;
    
    % compute prox-step for SAD* with Moreau's identity (if requested)
    if nargout == 2
        [~, prox] = SAD(L / sigma, I, 1 / sigma, false);
        res2 = L - sigma * prox;
    else
        res2 = [];
    end
    
end

end