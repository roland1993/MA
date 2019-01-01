function [res1, res2, res3] = ...
    nuclear_norm_constraint(L, numImg, tau, nu, conjugate_flag)
% IN:
%       L               ~ m*n*numImg x 1    all images in one column vector
%       numImg          ~ 1 x 1             number of images
%       tau             ~ 1 x 1             prox step size
%       nu              ~ 1 x 1             constraint threshold
%       conjugate_flag  ~ logical           eval. constraint or conjugate?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1             indicator of ||.||_* <= nu
%       res2            ~ m*n x 1           prox-step of indicator
%       res3            ~ 1 x 1             measure for hurt constraints
%   IF conjugate_flag:
%       res1            ~ 1 x 1             convex conjugate of indicator
%       res2            ~ m*n x 1           prox-step of conjugate
%       res3            ~ 1 x 1             measure for hurt constraints

% by default: evaluate constraint instead of its conjugate
if nargin < 5, conjugate_flag = false; end

% reshape L back into a matrix ~ m*n x numImg
L = reshape(L, [], numImg);

% intialize error measure with zero
res3 = 0;

if ~conjugate_flag
    
    % compute svd of L
    [~, S, ~] = svd(L, 'econ');
    S = diag(S);
    
    % return 0 as fctn. value of indicator
    res1 = 0;
    
    % distance to feasible region
    if sum(S) > nu
        res3 = sum(S) - nu;
    end
    
    % get prox operator via Moreau's identity (if requested)
    if nargout == 2
        [~, conj_prox] = ...
            nuclear_norm_constraint(L(:) / tau, numImg, 1 / tau, nu, true);
        res2 = L(:) - tau * conj_prox;
    else
        res2 = [];
    end
    
else
    
    % get SVD of L
    [U, S, V] = svd(L, 'econ');
    S = diag(S);
    
    % conjugate of nn-constraint = spectral norm = max singular value
    res1 = nu * max(S);
    
    % compute prox of spectral norm via prox of inf-norm of sv-vector S
    if nargout == 2
        
        % nu and tau in one factor
        mu = nu * tau;
        
        % prox on S via Moreau's identity
        %   -> conjugate of inf-norm is l1-ball indicator
        S_prox = S - mu * l1ball_projection(S / mu);
        
        % use prox step of sv-vector to compute prox of spectral norm
        res2 = U * diag(S_prox) * V';
        res2 = res2(:);
        
    else
        res2 = [];
    end
    
end

end