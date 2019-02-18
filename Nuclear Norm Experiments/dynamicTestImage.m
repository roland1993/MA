function T = dynamicTestImage(m, n, numFrames)

if ~exist('m', 'var'), m = 200; end
if ~exist('n', 'var'), n = 200; end
if ~exist('numFrames', 'var'), numFrames = 6; end

[xx, yy] = meshgrid(linspace(-1, 1, n), linspace(-1, 1, m));
T = zeros(m, n, numFrames);

f1 = 4 * pi;
p1 = 0.25 * (1 + rand) * pi;
f2 = 6 * pi;
p2 = 0.25 * rand * pi;

for i = 1 : numFrames
    
    dx = -0.1 * sin(pi * (i) / numFrames);
    dy = 0.1 * cos(pi * (i) / numFrames);

%     dx = 0.1 * (rand - 0.5);
%     dy = 0.2 * (rand - 0.5);
    
    ellipse_rad = 0.4;
    ellipse = double( ...
        sqrt(2 * (xx + dx - 0.25) .^ 2 + (yy + dy) .^ 2) <= ellipse_rad);
    if mod(i, 2) == 0
        texture = sin(f1 * (yy + dy) + p1) .^ 2;
    else
        texture = sin(f2 * (xx + dx) + p2) .^ 2;
    end
    IDX = sqrt(2 * (xx + dx - 0.25) .^ 2 + (yy + dy) .^ 2) ...
            <= 0.6 * ellipse_rad;
    ellipse(IDX) = texture(IDX);
    
    % frame_rad = 0.7 + 0.05 * (i / numFrames);
    frame_rad = 0.7;
    frame_width = 0.15;
    % tmp = 0.9 + 0.1 * (i / numFrames);
    tmp = 1;
    frame = double(reshape( ...
            frame_rad <= max(abs(tmp * [xx(:), yy(:)]), [], 2) & ...
            max(abs(tmp * [xx(:), yy(:)]), [], 2) <= ...
                                frame_rad + frame_width, m, n));
    
    rect = double(reshape((-0.6 <= xx(:)) & (xx(:) <= -0.2) & ...
        (-0.5 <= yy(:)) & (yy(:) <= 0.5), m, n));
    
    T(:, :, i) = imgaussfilt(rect + ellipse + frame, (m + n) / 150);
    
end

% figure;
% colormap gray(256);
% for i = 1 : numFrames
%     
%     imagesc(T(:, :, i));
%     axis image;
%     colorbar;
%     drawnow;
%     pause(1 / numFrames);
%     
% end

end