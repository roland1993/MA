function [f, df, d2f] = curvature_energy(u, s, h)
% IN:
%   u   ~ (m*n) x 2             displacement field to regularize
%   s   ~ 2 x 1                 s = [m, n]
%   h   ~ 2 x 1                 grid width
% OUT:
%   f   ~ 1 x 1                 curvature energy of u
%   df  ~ (m*n*2) x 1           gradient of curvature w.r.t u
%   d2f ~ (m*n*2) x (m*n*2)     Hessian of curvature w.r.t u

% make sure u has the right format...
u = reshape(u,[],2);

persistent L;
persistent LL;

% compute discrete Laplace L only once (for every resolution level)
if isempty(L) || size(L, 2) ~= numel(u)
    
    m = s(1);   n = s(2);
    
    % create discrete Laplacian operator
    %   i)  2nd order x-derivative
    e_x = ones(n,1);
    D_xx = (1 / h(1)) ^ 2 * spdiags([e_x, -2*e_x, e_x], -1 : 1, n, n);
    
    %   ii) 2nd order y-derivative
    e_y = ones(m, 1);
    D_yy = (1 / h(2)) ^ 2 * spdiags([e_y, -2*e_y, e_y], -1 : 1, m, m);
    
    % include boundary condition for cell-centered data
    bc = 'dn';
    switch bc
        case 'dn'
            % cell-centered dirichlet-null
            D_xx(1  ,1  ) = -1 / h(1)^2;
            D_xx(end,end) = -1 / h(1)^2;
            D_yy(1  ,1  ) = -1 / h(2)^2;
            D_yy(end,end) = -1 / h(2)^2;
        case 'nn'
            % cell-centered neumann-null
            D_xx(1  ,1  ) = -3 / h(1)^2;
            D_xx(end,end) = -3 / h(1)^2;
            D_yy(1  ,1  ) = -3 / h(2)^2;
            D_yy(end,end) = -3 / h(2)^2;
    end
    
    %   iii) combine to form discrete Laplacian
    L = kron(D_xx, speye(m)) + kron(speye(n), D_yy);
    L = kron(speye(2), L);
    
end

% compute curvature energy as (h1*h2)/2 * u(:)' * L' * L '* u(:)
l_u = L * u(:);
f = 0.5 * prod(h) * (l_u' * l_u);

% compute df/du as (h1*h2) * L' * L * u(:), d2f/du^2 as (h1*h2) * L' * L
if nargout >= 2
    df = prod(h) * L' * l_u;
end
if nargout == 3
    if isempty(LL) || size(LL, 2) ~= numel(u)
        LL = prod(h) * (L' * L);
    end
    d2f = LL;
end

end