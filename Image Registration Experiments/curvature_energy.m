function [f, df, d2f] = curvature_energy(u, s, h)
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
%   u   ~ (m*n) x 2             displacement field to regularize
%   s   ~ 2 x 1                 s = [m, n]
%   h   ~ 2 x 1                 grid width
% OUT:
%   f   ~ 1 x 1                 curvature energy of u
%   df  ~ (m*n*2) x 1           gradient of curvature w.r.t u
%   d2f ~ (m*n*2) x (m*n*2)     Hessian of curvature w.r.t u
%--------------------------------------------------------------------------

% make sure u has the right format...
u = reshape(u,[],2);

persistent LtL_h;

% compute discrete Laplace L only once (for every resolution level)
if isempty(LtL_h) || size(LtL_h, 2) ~= numel(u)
    
    m = s(1);   n = s(2);
    
    % create discrete Laplacian operator
    %   i)  2nd order x-derivative
    e_x = ones(m, 1);
    D_xx = (1 / h(1)) ^ 2 * spdiags([e_x, -2*e_x, e_x], -1 : 1, m, m);
    
    %   ii) 2nd order y-derivative
    e_y = ones(n, 1);
    D_yy = (1 / h(2)) ^ 2 * spdiags([e_y, -2*e_y, e_y], -1 : 1, n, n);
    
    % include boundary condition for cell-centered data
    bc = 'nn';
    switch bc
        case 'nn'
            % cell-centered neumann-null
            D_xx(1  ,1  ) = -1 / h(1)^2;
            D_xx(end,end) = -1 / h(1)^2;
            D_yy(1  ,1  ) = -1 / h(2)^2;
            D_yy(end,end) = -1 / h(2)^2;
        case 'dn'
            % cell-centered dirichlet-null
            D_xx(1  ,1  ) = -3 / h(1)^2;
            D_xx(end,end) = -3 / h(1)^2;
            D_yy(1  ,1  ) = -3 / h(2)^2;
            D_yy(end,end) = -3 / h(2)^2;
    end
    
    %   iii) combine to form discrete Laplacian L
    L = kron(speye(n), D_xx) + kron(D_yy, speye(m));
    L = kron(speye(2), L);
    
    % pre-compute and save h(1)*h(2)*L'*L (necessary for computing output)
    LtL_h = prod(h) * (L' * L);
    
end

d2f = LtL_h;                % d2f = prod(h) * L' * L
df = LtL_h * u(:);          % df  = prod(h) * L' * L * u
f = 0.5 * u(:)' * df;       % f   = 0.5 * prod(h) * u' * L' * L * u

end