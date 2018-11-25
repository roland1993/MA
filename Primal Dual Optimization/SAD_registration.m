function [res1, res2] = ...
    SAD_registration(u, u0, T, R, h, tau, conjugate_flag)
% IN:
%       u               ~ m*n*2 x 1     evaluation point
%       u0              ~ m*n*2 x 1     development point for linearizing T
%       T               ~ m x n         original template image
%       R               ~ m x n         reference image
%       h               ~ 2 x 1         grid spacing
%       tau             ~ 1 x 1         prox-operator step size
%       conjugate_flag  ~ logical       evaluate SAD or SAD* at u?
% OUT:
%   IF NOT conjugate_flag:
%       res1            ~ 1 x 1         function value SAD(u)
%       res2            ~ m*n*2 x 1     prox step of SAD at u
%   IF conjugate_flag:
%       res1            ~ 1 x 1         convex conjugate value SAD*(u)
%       res2            ~ m*n*2 x 1     prox step of SAD* at u

% by default: evaluate SAD instead of its conjugate
if nargin < 7, conjugate_flag = false; end

% linearize interpolation of T at u0
[T, grad_T] = evaluate_displacement(T, h, reshape(u0, [], 2));

if ~conjugate_flag
    % EITHER ~> evaluate SAD and Prox_[SAD] at u
    
    % SAD(u) = ||T(u0) + grad_T(u0) * (u - u0) - R||_1
    phi = (T(:) + grad_T * (u - u0)) - R(:);
    res1 = sum(abs(phi));
    
    if nargout == 2
        
        % reshape sparse grad_T ~ m*n x m*n*2 to ~ m*n x 2
        k = size(grad_T, 1);                            % get k = m * n
        grad_T = [diag(grad_T, 0), diag(grad_T, k)];
        norm_grad_squared = sum(grad_T .^ 2, 2);
        
        % Prox_[SAD] at u = casewise thresholding step
        %   -> get indices of different cases
        idx1 = (phi < (-tau) * norm_grad_squared);
        idx2 = (phi > tau * norm_grad_squared);
        idx3 = ~(idx1 | idx2);
        
        % split case 3 to prevent division by zero!
        idx3_1 = idx3 & (norm_grad_squared > 0);
        idx3_2 = idx3 & (norm_grad_squared == 0);
        
        u = reshape(u, k, 2);
        res2 = zeros(size(u));
        
        % case 1
        res2(idx1, :) = u(idx1, :) + tau * grad_T(idx1, :);
        % case 2
        res2(idx2, :) = u(idx2, :) - tau * grad_T(idx2, :);
        % case 3
        res2(idx3_1, :) = u(idx3_1, :) - grad_T(idx3_1, :) .* ...
            (phi(idx3_1) ./ norm_grad_squared(idx3_1));
        res2(idx3_2, :) = u(idx3_2, :);
        
        res2 = res2(:);
        
    end
    
else
    % ...todo
    res1 = -inf;
    if nargout == 2, res2 = -inf * ones(size(u0)); end
end

end