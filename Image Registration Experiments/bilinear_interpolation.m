function X_p = bilinear_interpolation(X, h, p)
% IN:
%   X ~ m x n       image
%   h ~ 2 x 1       grid width [h_x, h_y]
%   p ~ l x 2       evaluation points
% OUT:
%   X_p ~ l x 1     bilinear interpolation of X at points p

[m, n] = size(X);
l = size(p, 1);

% transform (x, y)-coordinates to (i, j)-coordinates
q = p - (h / 2);
r = q ./ h;
s = [r(:, 1), -r(:, 2) + (m - 1)];
t = [s(:, 2), s(:, 1)];
u = t + 1;

% pad X with zeros for boundary condition
Y = padarray(X, [1 1]);

% evaluate linear interpolation of X at given points p
X_p = zeros(l, 1);
for i = 1 : l
    
    % set interpolation value to 0 outside of [0, m] x [0, n]
    if (u(i, 1) <= 0) || (u(i, 1) >= (m + 1)) || ...
            (u(i, 2) <= 0) || (u(i, 2) >= (n + 1))
        X_p(i) = 0;
    else
        int = floor(u(i, :));
        dec = u(i, :) - int;
        
        X_p(i) = ...
            (1 - dec(1)) * (1 - dec(2)) * Y(int(1) + 1, int(2) + 1) + ...
            dec(1) * (1 - dec(2)) * Y(int(1) + 2, int(2) + 1) + ...
            (1 - dec(1)) * dec(2) * Y(int(1) + 1, int(2) + 2) + ...
            dec(1) * dec(2) * Y(int(1) + 2, int(2) + 2);
    end
    
end

end