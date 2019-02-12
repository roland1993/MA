function [res1, res2, res3] = ...
    nuclear_norm_constraint_mod(L, d, numImg, tau, nu, conjugate_flag)
% IN:
%       L               ~ m*n*numImg x 1    input variables
%       d               ~ m*n*numImg x 1    constant offset from input
%       numImg          ~ 1 x 1             number of images in L
%       tau             ~ 1 x 1             prox step size
%       nu              ~ 1 x 1             constraint threshold
%       conjugate_flag  ~ logical           eval. constraint or conjugate?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1             delta_{||.||_* <= nu}(L - R)
%       res2            ~ 1 x 1             constraint violation measure
%       res3            ~ m*n x 1           prox-step of indicator
%   IF conjugate_flag:
%       res1            ~ 1 x 1             weighted spectral norm
%       res2            ~ 1 x 1             constraint violation measure
%       res3            ~ m*n x 1           prox-step of spectral norm

% IMPLEMENTATION OF delta_{|| . ||_* <= nu}(L - d)

% by default: evaluate constraint instead of its conjugate
if nargin < 6, conjugate_flag = false; end

% intialize error measure with zero
res2 = 0;

if ~conjugate_flag
    
    % subtract d from L
    L = L - d;
    
    % reshape L back into a matrix ~ m*n x numImg
    L = reshape(L, [], numImg);
    
    % compute svd of L
    [U, S, V] = svd(L, 'econ');
    S = diag(S);
    
    % return 0 as fctn. value of indicator
    res1 = 0;
    
    % distance to feasible region
    if sum(S) > nu
        res2 = sum(S) - nu;
    end
    
    % get prox operator 
    if nargout == 3
        
        % l1-ball-projection of SV-vector
        res3 = U * diag(nu * l1ball_projection(S / nu)) * V';
        
        % reshape to vector format and add d
        res3 = res3(:) + d;
        
    end
    
elseif conjugate_flag && nargout < 3
    
    % reshape L back into a matrix ~ m*n x numImg
    L = reshape(L, [], numImg);
    
    % get SVD of L
    [~, S, ~] = svd(L, 'econ');
    S = diag(S);
    
    % conjugate of nn-constraint = spectral norm = max singular value
    %   + <L, d>
    res1 = nu * max(S) + L(:)' * d;
    
else
    
    res1 = [];
    res2 = [];
    
    % compute conjugate prox via Moreau
    [~, ~, prox] = nuclear_norm_constraint_mod( ...
        L(:) / tau, d, numImg, 1 / tau, nu, false);
    res3 = L(:) - tau * prox;
    
end

end