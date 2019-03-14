function [res1, res2, res3] = ...
    SSD_registration(u, u0, T, R, h, lambda, tau, conjugate_flag)
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
%       u               ~ m*n*2 x 1     evaluation point
%       u0              ~ m*n*2 x 1     development point for linearizing T
%       T               ~ m x n         original template image
%       R               ~ m x n         reference image
%       h               ~ 2 x 1         grid spacing
%       lambda          ~ 1 x 1         weighting factor
%       tau             ~ 1 x 1         prox-operator step size
%       conjugate_flag  ~ logical       evaluate SSD or SSD* at u?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1         function value SSD(u)
%       res2            ~ 1 x 1         constraint violation measure
%       res3            ~ m*n*2 x 1     prox step of SSD at u
%   IF conjugate_flag:
%       res1            ~ 1 x 1         convex conjugate value SSD*(u)
%       res2            ~ 1 x 1         constraint violation measure
%       res3            ~ m*n*2 x 1     prox step of SSD* at u
%--------------------------------------------------------------------------

% by default: evaluate SAD instead of its conjugate
if nargin < 8, conjugate_flag = false; end

% initialize constraint measure with 0
res2 = 0;

% get k = m * n
k = numel(T);

% linearize interpolation of T at u0
[T, grad_T] = evaluate_displacement(T, h, reshape(u0, k, 2));
b = T(:) - (grad_T * u0) - R(:);

if ~conjugate_flag
    % EITHER ~> evaluate SSD and Prox_[SSD] at u
    
    % SSD = 0.5 * ||grad_T(u0) * u + b||^2, b = T(u0) - grad_T(u0)*u0 - R
    residual = (grad_T * u) + b;
    res1 = 0.5 * lambda * (residual' * residual);
    
    % Prox_[SSD]
    if nargout == 3
        
        D1 = grad_T(:, 1 : k);
        D2 = grad_T(:, (k + 1) : end);
        A = [speye(k) + (lambda * tau) * D1 .^ 2, ...
            (lambda * tau) * D1 .* D2; ...
            (lambda * tau) * D1 .* D2, ...
            speye(k) + (lambda * tau) * D2 .^ 2];
        c = u - (lambda * tau) * grad_T' * b;
        res3 = A \ c;
        
    end
    
else
    
    % scaling lambda ~> [lambda * G]*(u) = lambda * G*(u/lambda)
    u = u / lambda;
    
    % conjugate of SSD = sum_ij 0.5 * (grad_T_ij * u_ij + b_ij) ^ 2
    %   = sum_ij {0.5 * u_ij' * grad_T_ij' * grad_T_ij * u_ij
    %       + u_ij' * grad_T_ij' * b_ij
    %       + 0.5 * b_ij ^ 2}
    % computed as pointwise conjugates of a quadratic model
    %   0.5 * x' * A * x + x' * b + c
    % ~> see Rockafellar, page 481
    grad_T = spdiags(grad_T);
    norm_grad_squared = sum(grad_T .^ 2, 2);
    idx = (norm_grad_squared > 1e-7);
    
    u = reshape(u, [], 2);
    res1 = 0.5 * sum((u - b .* grad_T) .* grad_T, 2) .^ 2;
    res1(idx) = res1(idx) ./ (norm_grad_squared(idx) .^ 2);
    
    res1 = res1 - 0.5 * b .^ 2;
    res1 = lambda * sum(res1);
    
    % constraint: (v - b) has to be in Image(pinv(A))
    %   -> error measure given by distance from this subspace
    res2 = zeros(size(idx));
    
    % case 1: grad_T_ij = 0
    %   -> error measure is distance from 0
    res2(~idx) = sqrt(sum(u(~idx, :) .^ 2, 2));
    
    % case 2: grad_T_ij ~= 0
    %   -> error measure is distance to subspace {mu * grad_T_ij}
    grad_T(idx, :) = grad_T(idx, :) ./ sqrt(norm_grad_squared(idx));
    grad_T_orth = [-grad_T(:, 2), grad_T(:, 1)];
    res2(idx) = abs(sum(grad_T_orth(idx, :) .* u(idx, :), 2));
    
    % return maximum error over all ij
    res2 = max(res2);
    
    % Prox_[SSD*]
    if nargout == 3
        
        % compute prox-step for G* = SSD* with Moreau's identity
        %   [(id + tau * dG*)^(-1)](u) =
        %       u - tau * [(id + (1 / tau) * dG)^(-1)](u / tau)
        [~, ~, prox] = ...
            SSD_registration(u / (lambda * tau), u0, T, R, h, ...
            lambda, 1 / (lambda * tau), false);
        res3 = u - (lambda * tau) * prox;
        
    end
    
end

end