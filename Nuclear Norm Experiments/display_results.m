function display_results(img, refIdx, u, L)

% k = number of templates
k = length(img) - 1;

% get reference image
R = img{refIdx};

% some indexing
tempIdx = 1 : (k + 1);
tempIdx(refIdx) = [];
IDX = [tempIdx, refIdx];

% get image resolution etc.
[m, n] = size(R);
h_img = [1, 1];
omega = [0, m, 0, n];
h_grid = (omega([2, 4]) - omega([1, 3])) ./ [m, n];

% evaluate displacements
img_u = cell(k + 1, 1);
for i = 1 : k
    img_u{tempIdx(i)} = ...
        evaluate_displacement(img{tempIdx(i)}, h_img, u(:, :, i));
end
img_u{refIdx} = img{refIdx};

% get cc-grid for plotting
[cc_x, cc_y] = cell_centered_grid(omega, [m, n]);
cc_grid = [cc_x(:), cc_y(:)];

% get green image for visualizing image differences
green = cat(3, zeros(m, n), ones(m, n), zeros(m, n));

% get mean of L
meanL = sum(L, 3) / (k + 1);

% do the plotting
figure;
colormap gray(256);

for i = 1 : (k + 1)
    
    subplot(3, k + 1, i);
    imshow(img{IDX(i)}, [0 1], 'InitialMagnification', 'fit');
    if i <= k
        hold on;
        quiver(cc_grid(:, 2), cc_grid(:, 1), ...
            u(:, 2, i), u(:, 1, i), 0, 'r');
        hold off;
        title(sprintf('T_%d with u_%d', i, i));
    else
        title('R');
    end
    
    subplot(3, k + 1, (k + 1) + i);
    imshow(img_u{IDX(i)}, [0 1], 'InitialMagnification', 'fit');
    hold on;
    imagesc(...
        'YData', omega(1) + h_grid(1) * [0.5, m - 0.5], ...
        'XData', omega(3) + h_grid(2) * [0.5, n - 0.5], ...
        'CData', green, ...
        'AlphaData', abs(img_u{IDX(i)} - L(:, :, i)));
    hold off;
    if i <= k
        title(sprintf('T_%d(u_%d) with |T_%d(u_%d) - l_%d|', ...
            i, i, i, i, i));
    else
        title(sprintf('R with |R - l_%d|', i));
    end
    
    subplot(3, k + 1, 2 * (k + 1) + i);
    imshow(L(:, :, i) - meanL, [-1 1], 'InitialMagnification', 'fit');
    title(sprintf('l_%d - l_{mean}', i, i));
    
end

end