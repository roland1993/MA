function [f, df] = SSD(T, R, h, u)
% IN:
%   T ~ m x n           template image
%   R ~ m x n           reference image
%   h ~ 2 x 1           grid width
%   u ~ (m*n) x 2       displacement field [u_x, u_y] for T
% OUT:
%   f  ~ 1 x 1          SSD ||T(u) - R|| ^ 2
%   df ~ (m*n*2) x 1    dSSD/du

% Was the gradient df/du requested?
if nargout == 2
    [T_u, dT_u] = evaluate_displacement(T, h, u);
else
    T_u = evaluate_displacement(T, h, u);
end

% compute SSD
f = 0.5 * sum((T_u(:) - R(:)) .^ 2);

% compute gradient df/du
if nargout == 2
    
    % gradient of outer fctn. phi(x) = ||x|| ^ 2
    dphi = T_u - R;
    
    % gradient of inner function (T(u) - R) is dT_u
    %   -> df = dphi * dT_u
    df = dphi .* dT_u;
    
    % reshape df to
    %   [df/du_11_x, .., df/du_mn_x, df/du_11_y, .., df/du_mn_y]'
    df = df(:);
    
end

end