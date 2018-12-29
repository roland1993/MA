%% 3D (FOR TESTING PROJECTION)

clear all, close all, clc;

[X, Y, Z] = ndgrid(-1 : 0.1 : 1, -1 : 0.1 : 1, -1 : 0.1 : 1);
P = [X(:), Y(:), Z(:)];
idx = sum(abs(P), 2) <= 1;
figure;
scatter3(P(idx, 1), P(idx, 2), P(idx, 3), 'r.');
axis equal;

Q = [...
    +1,  0,  0; ...
    -1,  0,  0; ...
    0, +1,  0; ...
    0, -1,  0; ...
    0,  0, +1; ...
    0,  0, -1; ...
    ];

L = [...
    1, 3, 5; ...
    1, 3, 6; ...
    1, 4, 5; ...
    1, 4, 6; ...
    2, 3, 5; ...
    2, 3, 6; ...
    2, 4, 5; ...
    2, 4, 6; ...
    ];

q = rand(3, 1);
r = l1ball_projection(q);

hold on;
for i = 1 : size(L, 1)
    tmp = Q([L(i, :), L(i, 1)], :);
    plot3(tmp(:, 1), tmp(:, 2), tmp(:, 3), 'k', 'LineWidth', 2);
end
plot3([q(1), r(1)], [q(2), r(2)], [q(3), r(3)], 'b-o');
hold off;

%% ND (FOR TESTING TIME COMPLEXITY)

clear all, close all, clc;

N = 5 * 10 .^ (0 : 7);
T = zeros(size(N));

for n = 1 : numel(N)
    for i = 1 : 10
        q = randn(1, N(n));
        q = 1.1 * q / sum(abs(q));
        tic;
        l1ball_projection(q);
        T(n) = T(n) + toc;
    end
    T(n) = T(n) / 10;
end

figure;
loglog(N, T, '-o');
grid on;