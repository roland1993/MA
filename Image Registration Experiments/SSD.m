function [f, df, d2f] = SSD(T, R, h, u)
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
%   T   ~ m x n                 template image
%   R   ~ m x n                 reference image
%   h   ~ 2 x 1                 grid width
%   u   ~ (m*n) x 2             displacement field [u_x, u_y] for T
% OUT:
%   f   ~ 1 x 1                 SSD ||T(u) - R|| ^ 2
%   df  ~ (m*n*2) x 1           gradient dSSD/du
%   d2f ~ (m*n*2) x (m*n*2)     approximate Hesssian of SSD (see line 42f.)
%--------------------------------------------------------------------------

% make sure u has the right format...
u = reshape(u, [], 2);

% Was the gradient df/du requested?
if nargout >= 2
    [T_u, dT_u] = evaluate_displacement(T, h, u);
else
    T_u = evaluate_displacement(T, h, u);
end

% compute SSD
f = 0.5 * prod(h) * sum((T_u(:) - R(:)) .^ 2);

% compute gradient df/du
if nargout >= 2
    
    % gradient of outer fctn. phi(x) = ||x|| ^ 2
    dphi = T_u - R;
    
    % gradient of inner function (T(u) - R) is dT_u
    %   -> df = dphi * dT_u
    df = prod(h) * (dphi(:)' * dT_u)';
    
    % approximate Hessian of SSD in the sense, that 
    %   SSD(u+r) = ||T(u+r) - R||^2 is approximated by
    % 	Q(u+r) = ||(T(u)+dT(u)*r) - R||^2, for which the Hessian 
    %   is given by dT(u)'*dT(u) (regardless of r)
    %   ~~> necessary for Gauss-Newton optimization
    if nargout == 3
        d2f = prod(h) * (dT_u' * dT_u);
    end
    
end

end