function [res1, res2, res3] = ...
    SAD_registration(u, u0, T, R, h, lambda, tau, conjugate_flag)
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
%       res3            ~ 1 x 1         measure for hurt constraints
%   IF conjugate_flag:
%       res1            ~ 1 x 1         convex conjugate value SAD*(u)
%       res2            ~ m*n*2 x 1     prox step of SAD* at u
%       res3            ~ 1 x 1         measure for hurt constraints

% by default: evaluate SAD instead of its conjugate
if nargin < 8, conjugate_flag = false; end

% initialize measure for hurt constraints with 0
res3 = 0;

% linearize interpolation of T at u0
[T, grad_T] = evaluate_displacement(T, h, reshape(u0, [], 2));
b = T(:) - grad_T * u0 - R(:);

if ~conjugate_flag
    % EITHER ~> evaluate SAD and Prox_[SAD] at u
    
    % SAD(u) = ||T(u0) + grad_T(u0) * (u - u0) - R||_1
    phi = b + grad_T * u;
    res1 = lambda * sum(abs(phi));
    
    if nargout == 2
        
        % reshape sparse grad_T ~ m*n x m*n*2 to ~ m*n x 2
        k = size(grad_T, 1);            % get k = m * n
        grad_T = [diag(grad_T, 0), diag(grad_T, k)];
        norm_grad_squared = sum(grad_T .^ 2, 2);
        
        % Prox_[SAD] at u = casewise thresholding step
        %   -> get indices of different cases
        idx1 = (phi < (-tau) * norm_grad_squared);
        idx2 = (phi > tau * norm_grad_squared);
        idx3 = ~(idx1 | idx2);
        
        % split case 3 to prevent division by zero!
        idx3_1 = idx3 & (norm_grad_squared > 1e-14);
        idx3_2 = idx3 & (norm_grad_squared <= 1e-14);
        
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
        
    else
        res2 = [];
    end
    
else
    % OR ~> evaluate SAD* and Prox_[SAD*] at u
    
    % [lambda * G(#)]* = lambda * G(#/lambda)
    u = u / lambda;
    
    % conjugate is decoupled -> can be evaluated pointwise for every ij,
    %   which is the conjugate of G(u_ij) = abs(<grad_ij, u_ij> + b_ij)
    res1 = zeros(numel(T), 1);
    
    % evaluate constraint measure pointwise as well
    res3 = zeros(numel(T), 1);
    
    % reshape u and gradient to format ~ m*n x 2
    u = reshape(u, [], 2);
    grad_T = spdiags(grad_T);
    norm_grad_squared = sum(grad_T .^ 2, 2);      % get gradient size
    
    % case 1: gradient = 0, feasible region of SAD* is FR = {0}
    idx1 = (norm_grad_squared < 1e-14);
    
    % G*(u_ij) = -abs(b_ij)
    res1(idx1) = (-1) * abs(b(idx1));
    
    % dist(u_ij, FR) = ||u_ij||_2
    res3(idx1) = sqrt(sum(u(idx1, :) .^ 2, 2));
    
    % case 2: gradient ~= 0, FR = {t * grad_ij | -1 <= t <= 1}
    idx2 = ~idx1;
    
    % G*(u_ij) = <u_ij, p_ij>, where <grad_ij, p_ij> + b_ij = 0 for p_ij
    %   -> get p_ij as (-b_ij / <grad_ij, grad_ij>) * grad_ij
    p = (-b(idx2) ./ norm_grad_squared(idx2)) .* grad_T(idx2, :);
    res1(idx2) = sum(u(idx2, :) .* p, 2);
    
    % find dist(u_ij, FR) by rotating coordinate system
    %   -> align grad_ij with x-axis
    norm_grad = sqrt(norm_grad_squared);
    u_rot = [sum([grad_T(:, 1), grad_T(:, 2)] .* u, 2), ...
        sum([-grad_T(:, 2), grad_T(:, 1)] .* u, 2)];
    u_rot(idx2, :) = u_rot(idx2, :) ./ norm_grad(idx2);
    
    % case 2.1: u_rot_ij(1) > ||g_ij||_2
    idx2_1 = idx2 & (u_rot(:, 1) > norm_grad);
    res3(idx2_1) = sum((u(idx2_1, :) - grad_T(idx2_1, :)) .^ 2, 2);
    
    % case 2.2: u_rot_ij(1) < - ||g_ij||_2
    idx2_2 = idx2 & (u_rot(:, 1) < - norm_grad);
    res3(idx2_2) = sum((u(idx2_2, :) + grad_T(idx2_2, :)) .^ 2, 2);
    
    % case 2.3: |u_rot_ij(1)| <= ||g_ij||_2
    idx2_3 = idx2 & ~(idx2_1 | idx2_2);
    res3(idx2_3) = abs(u_rot(idx2_3, 2));
    
    % combine pointwise results to form output
    res1 = lambda * sum(res1);
    res3 = max(res3);
    
    % Prox_[SAD*]
    if nargout == 2
        
        % compute prox-step for G* = SAD* with Moreau's identity
        %   [(id + tau * dG*)^(-1)](u) =
        %       u - tau * [(id + (1 / tau) * dG)^(-1)](u / tau)
        [~, prox] = SAD_registration(u / (lambda * tau), u0, T, R, h, ...
            lambda, 1 / (lambda * tau), false);
        res2 = u - (lambda * tau) * prox;
        
    else
        res2 = [];
    end
    
end

end