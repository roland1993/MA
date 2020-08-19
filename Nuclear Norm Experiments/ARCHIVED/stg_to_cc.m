function S = stg_to_cc(m, n, dir)
% dir = 1   <->     grid staggered in vertical direction ~ (m + 1) x n
% dir = 2   <->     grid staggered in horizontal direction ~ m x (n + 1)

if dir == 2
    
    S = spdiags(0.5 * ones(n, 2), 0 : 1, n, n + 1);
    S = kron(S, speye(m));
    
elseif dir == 1
    
    S = spdiags(0.5 * ones(m, 2), 0 : 1, m, m + 1);
    S = kron(speye(n), S);
    
end

end