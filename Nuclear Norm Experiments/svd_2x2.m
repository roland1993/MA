function [U, Sigma, V] = svd_2x2(M)
% compute the 2x2-svd of each row of M ~ (k x 4)
%               [   |       |       |       |       ]
%   I.e., M =   [   m11     m21     m12     m22     ]
%               [   |       |       |       |       ]
%   with M = U * DIAG(Sigma) * V' (per row)

M = reshape(M, [], 4);

a = M(:, 1);
c = M(:, 2);
b = M(:, 3);
d = M(:, 4);

s1 = a .^ 2 + b .^ 2 + c .^ 2 + d .^ 2;
s2 = sqrt((a .^ 2 + b .^ 2 - c .^ 2 - d .^ 2) .^ 2 + 4 * (a .* c + b .* d) .^ 2);

% singular values of M
sigma1 = sqrt(max((s1 + s2) / 2, 0));
sigma2 = sqrt(max((s1 - s2) / 2, 0));

theta = atan2(2 * a .* c + 2 * b .* d, a .^ 2 + b .^ 2 - c .^ 2 - d .^ 2) / 2;

% left singular vectors of M
u11 = cos(theta);
u21 = sin(theta);
u12 = -sin(theta);
u22 = cos(theta);

phi = atan2(2 * a .* b + 2 * c .* d, a .^ 2 - b .^ 2 + c .^ 2 - d .^ 2) / 2;
c_phi = cos(phi);
s_phi = sin(phi);

s11 = (a .* u11 + c .* u21) .* c_phi + (b .* u11 + d .* u21) .* s_phi;
s22 = (a .* u21 - c .* u11) .* s_phi + (-b .* u21 + d .* u11) .* c_phi;
sign1 = sign(s11);
sign2 = sign(s22);

% right singular vectors of M
v11 = sign1 .* c_phi;
v21 = sign1 .* s_phi;
v12 = -sign2 .* s_phi;
v22 = sign2 .* c_phi;

U = [u11, u21, u12, u22];
Sigma = [sigma1, sigma2];
V = [v11, v21, v12, v22];

end


% To reconstruct M, use
%   M_11 = Sigma(:, 1) .* U(:, 1) .* V(:, 1) + Sigma(:, 2) .* U(:, 3) .* V(:, 3);
%   M_21 = Sigma(:, 1) .* U(:, 2) .* V(:, 1) + Sigma(:, 2) .* U(:, 4) .* V(:, 3);
%   M_12 = Sigma(:, 1) .* U(:, 1) .* V(:, 2) + Sigma(:, 2) .* U(:, 3) .* V(:, 4);
%   M_22 = Sigma(:, 1) .* U(:, 2) .* V(:, 2) + Sigma(:, 2) .* U(:, 4) .* V(:, 4);