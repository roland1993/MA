function [res1, res2, res3] = mean_zero_indicator_stg(u, s, conjugate_flag)

% fetch images dimensions etc.
m = s(1);
n = s(2);
k = s(3);

% size of x-/y-staggered grids
sz_stg_x = (m + 1) * n;
sz_stg_y = m * (n + 1);

% seperate x- and y-components
x_idx = repmat([true(sz_stg_x, 1); false(sz_stg_y, 1)], [k, 1]);
u_x = u(x_idx);
y_idx = repmat([false(sz_stg_x, 1); true(sz_stg_y, 1)], [k, 1]);
u_y = u(y_idx);

% normal-vector of mean-zero subspace
r1 = ones(sz_stg_x * k, 1);
norm_r1_squared = sz_stg_x * k;

r2 = ones(sz_stg_y * k, 1);
norm_r2_squared = sz_stg_y * k;

if ~conjugate_flag
    
    if nargout == 3
        
        % initialize prox
        res3 = zeros(size(u));
        
        % projection of u_x to subspace r' * v = 0    <=>   u_x-mean = 0
        res3(x_idx) = u_x - ((r1' * u_x) / norm_r1_squared) * r1;
        
        % projection of u_y to subspace r' * v = 0    <=>   u_y-mean = 0
        res3(y_idx) = u_y - ((r2' * u_y) / norm_r2_squared) * r2;
        
        % dummy outputs
        res1 = [];
        res2 = [];
        
    else
        
        % fctn. value for indicator
        res1 = 0;
        
        % distance from mean = 0
        res2 = max(abs([mean(u_x), mean(u_y)]));
        
    end
    
else
    
    % fctn. value for indicator
    res1 = 0;
    
    % initialize prox
    res3 = zeros(size(u));
    
    % projection of u_x to subspace span{r}
    res3(x_idx) = ((r1' * u_x) / norm_r1_squared) * r1;
    
    % projection of u_y to subspace span{r}
    res3(y_idx) = ((r2' * u_x) / norm_r2_squared) * r2;
    
    % constraint measure: distance to span{r}
%     res2 = max([norm(u_x - res3(x_idx)) / norm(u_x); ...
%         norm(u_y - res3(y_idx)) / norm(u_y)]);
    res2 = max([sum(abs(u_x - res3(x_idx))) / numel(u_x); sum(abs(u_y - res3(y_idx))) / numel(u_y)]);
    
end

end