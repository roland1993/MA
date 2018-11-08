function derivative_test(f, u, order)
% IN:
%   f       ~ function handle   target function, returns
%                                   - f: N x 1 -> 1 x 1 function value
%                                   - g: N x 1 -> N x 1 gradient vector
%                                   - H: N x 1 -> N x N Hessian matrix
%   u       ~ N x 1             evaluation point
%   order   ~ N x 1             highest order derivative to test (2 at max)

% set standard parameter
if nargin == 2
    order = 2;
end

% set step size and direction to test (randomized)
h = 10 .^ (0 : -1 : -10);
v = rand(size(u));      v = v / norm(v);

% evaluate f, g and H at u
if order == 0
    f_u = f(u);
    g_u = zeros(size(u));       % dummy
    H_u = zeros(numel(u));      % dummy
elseif order == 1
    [f_u, g_u] = f(u);
    H_u = zeros(numel(u));      % dummy
else
    [f_u, g_u, H_u] = f(u);
end

% evaluate f(u + h * v) to establish "ground truth" (for selected h-values)
f_exact = zeros(1, numel(h));
for i = 1 : numel(h)
    f_exact(i) = f(u + h(i) * v);
end

% approximate f by zero, first, second order Taylor polynomial
f_approx = zeros(order + 1, numel(h));
for i = 0 : order
    for j = 1 : numel(h)
        f_approx(i + 1, j) = ...
            f_u + ...
            (i >= 1) * h(j) * v' * g_u + ...
            (i == 2) * h(j) ^ 2 * 0.5 * v' * H_u * v;
    end
end

% calculate errors of different approximations
ERR = abs(repmat(f_exact, [order + 1, 1]) - f_approx);

% generate output
s1 = sprintf(...
    'Testing derivatives of %s with Taylor polynomials', func2str(f));
if order == 0
    s2 = sprintf(...
        '  h\t\t|{f-T_0}(u+h*v)|');
elseif order == 1
    s2 = sprintf(...
        '  h\t\t|{f-T_0}(u+h*v)|\t|{f-T_1}(u+h*v)|');
else
    s2 = sprintf(...
        '  h\t\t|{f-T_0}(u+h*v)|\t|{f-T_1}(u+h*v)|\t|{f-T_2}(u+h*v)|');
end
fprintf('\n%s\n%s\n\n%s\n\n', ...
    s1, ...
    repmat('-', [1, length(s1)]), ...
    s2);

for i = 1 : numel(h)
    fprintf('%1.0e', h(i));
    for j = 1 : (order + 1)
        fprintf('\t\t   %.4e', ERR(j, i));
        if j == (order + 1), fprintf('\n'), end
    end
end

if order == 0
    s3 = sprintf('Expected:%sO(h^1)', ...
        repmat(' ', [1, 12]));
elseif order == 1
    s3 = sprintf('Expected:%sO(h^1)%sO(h^2)', ...
        repmat(' ', [1, 12]), ...
        repmat(' ', [1, 18]));
else
    s3 = sprintf('Expected:%sO(h^1)%sO(h^2)%sO(h^3)', ...
        repmat(' ', [1, 12]), ...
        repmat(' ', [1, 18]), ...
        repmat(' ', [1, 18]));
end
fprintf('\n%s\n\n', s3);

% do some plotting
figure;
for i = 1 : (order + 1)
    loglog(h, ERR(i, :)', '-o');      hold on;
end
grid on;    hold off;   axis equal;
xlabel('h');
title(...
    sprintf('logarithmic error plot for %s', ...
    strrep(func2str(f), '_', '\_')));

names = cell(order + 1, 1);
for i = 1 : (order + 1)
    names{i} = sprintf('|{f-T_%d}(u+hv)|', i - 1);
end
legend(names, 'Location', 'NorthWest');

end