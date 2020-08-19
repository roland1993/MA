function P = staggered_prolongation(m, n, dir)
% dir = 1   <->     grid staggered in vertical direction ~ (m + 1) x n
% dir = 2   <->     grid staggered in horizontal direction ~ m x (n + 1)
%   (assuming neumann boundary conditions)

if dir == 2
    
    I1 = [  kron(speye(m - 1), [0.75; 0.25]), sparse(2 * (m - 1), 1)] + ...
         [  sparse(2 * (m - 1), 1), kron(speye(m - 1), [0.25; 0.75])   ];
    I1 = [  sparse(1, 1, 1, 1, m); ...
            I1;
            sparse(1, m, 1, 1, m)   ];
    
    I2 = [  kron(speye(n), sparse([1; 0.5])), sparse(2 * n, 1) ] + ...
         [   sparse(2 * n, 1), kron(speye(n), [0; 0.5])];
    I2 = [  I2; ...
            sparse(1, n + 1, 1, 1, n + 1)   ];
    
    P = kron(I2, I1);
    
elseif dir == 1
    
    I1 = [  kron(speye(m), sparse([1; 0.5])), sparse(2 * m, 1) ] + ...
            [   sparse(2 * m, 1), kron(speye(m), [0; 0.5])];
    I1 = [  I1; ...
            sparse(1, m + 1, 1, 1, m + 1)   ];
    
    I2 = [  kron(speye(n - 1), [0.75; 0.25]), sparse(2 * (n - 1), 1)] + ...
         [  sparse(2 * (n - 1), 1), kron(speye(n - 1), [0.25; 0.75])   ];
    I2 = [  sparse(1, 1, 1, 1, n); ...
            I2;
            sparse(1, n, 1, 1, n)   ];
    
    P = kron(I2, I1);
        
end

end