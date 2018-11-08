function [f, df, d2f] = diffusive_energy(u, s, h)
% IN:
%   u   ~ (m*n) x 2             displacement field to regularize
%   s   ~ 2 x 1                 s = [m, n]
%   h   ~ 2 x 1                 grid width
% OUT:
%   f   ~ 1 x 1                 diffusive energy of u
%   df  ~ (m*n*2) x 1           gradient of diffusive energy w.r.t. u
%   d2f ~ (m*n*2) x (m*n*2)     Hessian of diffusive energy w.r.t u

% make sure u has the right format...
u = reshape(u, [], 2);

persistent G;

% compute operator G only once
if isempty(G) || size(G, 2) ~= numel(u)
    
    % get gradient operators for x and y-direction
    [G_x, G_y] = gradient_operator(s, h);
    
    % form new gradient operator to operate on u(:)
    G = kron(speye(2), [G_x; G_y]);
    
end

% apply gradient operator u(:)
g_u = G * u(:);

% compute diffusiv energy for u
f = 0.5 * prod(h) * (g_u' * g_u);

% compute df/du and d2f/du^2 if requested
if nargout >= 2
    df = prod(h) * G' * g_u;
end
if nargout == 3
    d2f = prod(h) * (G' * G);
end

end