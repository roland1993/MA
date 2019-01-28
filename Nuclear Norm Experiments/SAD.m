function [res1, res2, res3] = SAD(L, I, sigma, conjugate_flag)
% IN:
%       L               ~ m*n*numImg x 1    variable to
%       I               ~ m*n*numImg x 1    all images in one column vector
%       sigma           ~ 1 x 1             prox step size
%       conjugate_flag  ~ logical           evaluate SAD or conjugate NN*?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1             function value SAD(L)
%       res2            ~ 1 x 1             constraint violation measure
%       res3            ~ m*n x 1           prox-step of SAD for L
%   IF conjugate_flag:
%       res1            ~ 1 x 1             convex conjugate SAD*(L)
%       res2            ~ 1 x 1             constraint violation measure
%       res3            ~ m*n x 1           prox-step of SAD* for L

% by default: evaluate SAD instead of its conjugate
if nargin < 4, conjugate_flag = false; end

% initialize constraint measure with 0
res2 = 0;

if ~conjugate_flag
    % EITHER ~> evaluate SAD and Prox_[SAD] at L
    
    % compute SAD
    res1 = sum(abs(L - I));
    
    if nargout == 3
        
        % prox-step for ||L - I||_1 =: SAD ~> pointwise shrinkage
        res3 = zeros(size(L));
        diff_LI = L - I;
        idx1 = (diff_LI > sigma);
        idx2 = (diff_LI < (-1) * sigma);
        idx3 = ~(idx1 | idx2);
        res3(idx1) = L(idx1) - sigma;
        res3(idx2) = L(idx2) + sigma;
        res3(idx3) = I(idx3);

    end
    
else
    % OR ~> evaluate SAD* and Prox_[SAD*] at L
    
    % compute SAD*(L) = delta_{||.||_inf <= 1}(L) + <L,I>
    if max(abs(L)) > 1
        res2 = max(abs(L)) - 1;
    end
    res1 = L' * I;
    
    % compute prox-step for SAD* with Moreau's identity (if requested)
    if nargout == 3
        [~, ~, prox] = SAD(L / sigma, I, 1 / sigma, false);
        res3 = L - sigma * prox;
    end
    
end

end