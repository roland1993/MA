function D = finite_differences_staggered(m, n, h_grid)
% assumes:
%   1. component: grid staggered in vertical direction ~ (m + 1) x n
%   2. component: grid staggered in horizontal direction ~ m x (n + 1)

Dx = (1 / h_grid(1)) * spdiags([-1 1] .* ones(m, 2), 0 : 1, m, m + 1);
Dy = (1 / h_grid(2)) * spdiags([-1 1] .* ones(n, 2), 0 : 1, n, n + 1);

D = [kron(speye(n), Dx); kron(Dy, speye(m))];

end