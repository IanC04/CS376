function [centers] = showAccumulator(im, radius)
img = im;
thetas = 0:359;
[r, c] = size(img, [1 2]);

% Get image of edge using edge-detection algorithm
grayImg = im2gray(img);
edges = edge(grayImg, "canny");
% figure;
% imshow(edges);

% Create accumulator array and use polar coordinates
accumulator = zeros(r, c);

% Calculate Hough radii
for i = 1:r
    for j = 1:c
        if edges(i, j) == 1
            x = i - 1;
            y = j - 1;
            for theta = thetas
                x_circ = round(x - cosd(theta) * radius);
                y_circ = round(y - sind(theta) * radius);
                if not(any([x_circ < 1, y_circ < 1, x_circ > r, y_circ > c]))
                    accumulator(x_circ, y_circ) = accumulator(x_circ, y_circ) + 1;
                end
            end
        end
    end
end

% Calculated accumulator matrix and normalize to display
% figure;
% acc_normal = mat2gray(accumulator);
% imshow(acc_normal);

subplot(1, 3, 1);
imshow(img);
title("Original Image with circle radius: " + radius);

subplot(1, 3, 2);
imshow(mat2gray(accumulator));
title("Accumulator matrix Pre-processed");

% Get local maxima
localCenters = imregionalmax(accumulator);
accumulator = accumulator .* localCenters;

subplot(1, 3, 3);
imshow(mat2gray(accumulator));
title("Accumulator matrix Post-processed");
end