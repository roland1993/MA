function P = cc_prolongation(m, n)

% interepolation in vertical direction
I1 = [kron(speye(m - 1), [3/4; 1/4]), sparse(2 * (m - 1), 1)] + ...
        [sparse(2 * (m - 1), 1), kron(speye(m - 1), [1/4; 3/4])];
I1 = [sparse(1, 1, 3/4, 1, m); I1; sparse(1, m, 3/4, 1, m)];
    
% interpolation in horizontal direction
I2 = [kron(speye(n - 1), [3/4; 1/4]), sparse(2 * (n - 1), 1)] + ...
        [sparse(2 * (n - 1), 1), kron(speye(n - 1), [1/4; 3/4])];
I2 = [sparse(1, 1, 3/4, 1, n); I2; sparse(1, n, 3/4, 1, n)];

P = kron(I2, I1);

end