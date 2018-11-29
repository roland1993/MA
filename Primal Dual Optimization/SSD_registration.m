function [res1, res2, res3] = ...
    SSD_registration(u, u0, T, R, h, tau, conjugate_flag)
% IN:
%       u               ~ m*n*2 x 1     evaluation point
%       u0              ~ m*n*2 x 1     development point for linearizing T
%       T               ~ m x n         original template image
%       R               ~ m x n         reference image
%       h               ~ 2 x 1         grid spacing
%       tau             ~ 1 x 1         prox-operator step size
%       conjugate_flag  ~ logical       evaluate SSD or SSD* at u?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1         function value SSD(u)
%       res2            ~ m*n*2 x 1     prox step of SSD at u
%       res3            ~ 1 x 1         measure for hurt constraints
%   IF conjugate_flag:
%       res1            ~ 1 x 1         convex conjugate value SSD*(u)
%       res2            ~ m*n*2 x 1     prox step of SSD* at u
%       res3            ~ 1 x 1        	measure for hurt constraints

% by default: evaluate SAD instead of its conjugate
if nargin < 7, conjugate_flag = false; end

% initialize constraint measure with 0
res3 = 0;

% get k = m * n
k = numel(T);

% linearize interpolation of T at u0
[T, grad_T] = evaluate_displacement(T, h, reshape(u0, k, 2));
b = T(:) - (grad_T * u0) - R(:);

if ~conjugate_flag
    % EITHER ~> evaluate SSD and Prox_[SSD] at u
    
    % SSD = 0.5 * ||grad_T(u0) * u + b||^2, b = T(u0) - grad_T(u0)*u0 - R
    residual = (grad_T * u) + b;
    res1 = (1 / 2) * (residual' * residual);
    
    % Prox_[SSD]
    if nargout == 2
        
        D1 = grad_T(:, 1 : k);
        D2 = grad_T(:, (k + 1) : end);
        A = [speye(k) + tau * D1 .^ 2, tau * D1 .* D2; ...
            tau * D1 .* D2, speye(k) + tau * D2 .^ 2];
        c = u - tau * grad_T' * b;
        res2 = A \ c;
        
    else
        res2 = [];
    end
    
else
    
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
    res1 = sum(res1);
    
    % constraint: (v - b) has to be in Image(pinv(A))
    %   -> error measure given by distance from this subspace
    res3 = zeros(size(idx));
    
    % case 1: grad_T_ij = 0
    %   -> error measure is distance from 0
    res3(~idx) = sqrt(sum(u(~idx, :) .^ 2, 2));
    
    % case 2: grad_T_ij ~= 0
    %   -> error measure is distance to subspace {mu * grad_T_ij}
    grad_T(idx, :) = grad_T(idx, :) ./ sqrt(norm_grad_squared(idx));
    grad_T_orth = [-grad_T(:, 2), grad_T(:, 1)];
    res3(idx) = abs(sum(grad_T_orth(idx, :) .* u(idx, :), 2));
    
    % return maximum error over all ij
    res3 = max(res3);
    
    % Prox_[SSD*]
    if nargout == 2
        
        % compute prox-step for G* = SSD* with Moreau's identity
        %   [(id + tau * dG*)^(-1)](u) =
        %       u - tau * [(id + (1 / tau) * dG)^(-1)](u / tau)
        [~, prox] = SSD_registration(u / tau, u0, T, R, h, 1 / tau, false);
        res2 = u - tau * prox;
        
    else
        res2 = [];
    end
    
end

end