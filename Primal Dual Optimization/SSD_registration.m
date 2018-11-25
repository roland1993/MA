function [res1, res2] = ...
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
%   IF conjugate_flag:
%       res1            ~ 1 x 1         convex conjugate value SSD*(u)
%       res2            ~ m*n*2 x 1     prox step of SSD* at u

% by default: evaluate SAD instead of its conjugate
if nargin < 7, conjugate_flag = false; end

% get k = m * n
k = numel(T);

% linearize interpolation of T at u0
[T, grad_T] = evaluate_displacement(T, h, reshape(u0, k, 2));

if ~conjugate_flag
    % EITHER ~> evaluate SSD and Prox_[SSD] at u
    
    % SSD = 0.5 * ||grad_T(u0) * u + b||^2, b = T(u0) - grad_T(u0)*u0 - R
    b = T(:) - (grad_T * u0) - R(:);
    residual = (grad_T * u) + b;
    res1 = (1 / 2) * (residual' * residual);
    
    if nargout == 2
        
        D1 = grad_T(:, 1 : k);
        D2 = grad_T(:, (k + 1) : end);
        
        A = [speye(k) + tau * D1 .^ 2, tau * D1 .* D2; ...
            tau * D1 .* D2, speye(k) + tau * D2 .^ 2];
        
        c = u - tau * grad_T' * b;
        
        res2 = A \ c;
        
    end
    
else
    % ...todo
    res1 = -inf;
    if nargout == 2, res2 = -inf * ones(size(u0)); end
end

end