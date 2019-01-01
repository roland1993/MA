function [res1, res2, res3] = ...
    nuclear_norm(L, numImg, tau, mu, conjugate_flag)
% IN:
%       L               ~ m*n*numImg x 1    all images in one column vector
%       numImg          ~ 1 x 1             number of images
%       tau             ~ 1 x 1             prox step size
%       mu              ~ 1 x 1             weighting factor
%       conjugate_flag  ~ logical           evalulate NN or conjugate NN*?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1             function value NN(L)
%       res2            ~ m*n x 1           prox-step of NN for L
%       res3            ~ 1 x 1             measure for hurt constraints
%   IF conjugate_flag:
%       res1            ~ 1 x 1             convex conjugate NN*(L)
%       res2            ~ m*n x 1           prox-step of NN* for L
%       res3            ~ 1 x 1             measure for hurt constraints

% by default: evaluate nuclear norm instead of its conjugate
if nargin < 5, conjugate_flag = false; end

% reshape L back into a matrix ~ m*n x numImg
L = reshape(L, [], numImg);

% intialize error measure with zero
res3 = 0;

if ~conjugate_flag
    % EITHER ~> evaluate NN and Prox_[NN] at L, weighted by mu
    
    % compute svd of L
    [U, S, V] = svd(L, 'econ');
    S = diag(S);
    
    % nuclear norm = l1-norm of S = (sigma_1, .., sigma_p)'
    res1 = mu * norm(S, 1);
    
    if nargout == 2
        
        % Prox_[NN] by spectral soft thresholding on S
        S_threshold = max(S - mu * tau, 0);
        k = numel(S_threshold);
        res2 = U * spdiags(S_threshold, 0, k, k) * V';
        res2 = res2(:);
        
    else
        res2 = [];
    end
    
else
    % OR ~> evaluate NN* and Prox_[NN*] at L
    
    % [mu * f(.)]* = mu * f*(. / mu)
    L = L / mu;
    
    % compute svd of L
    [~, S, ~] = svd(L, 'econ');
    S = diag(S);
    
    % conjugate of NN = conjugate of 1-norm (on S)
    %   -> indicator d_{||.||_inf <= 1}(S)
    res1 = mu * 0;
    if max(S) > 1
        res3 = max(S) - 1;
    end
    
    % compute prox-step for NN* with Moreau's identity (if requested)
    if nargout == 2
        [~, prox] = nuclear_norm(L(:) / (mu * tau), numImg, ...
            1 / tau, 1 / mu, false);
        res2 = L(:) - mu * tau * prox;
    else
        res2 = [];
    end
    
end

end