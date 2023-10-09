function [centers] = detectCirclesHT(im, radius)
img = im;
thetas = 0:359;
[height, width] = size(img, [1 2]);
bin_size = uint32(1);
bin_amount_height = idivide(height, bin_size, "ceil");
bin_amount_width = idivide(width, bin_size, "ceil");

% Get image of edge using edge-detection algorithm
grayImg = im2gray(img);
edges = edge(grayImg, "canny");

% Create accumulator array and use polar coordinates
accumulator = zeros(bin_amount_height, bin_amount_width);

% Calculate Hough radii
for i = 1:height
    for j = 1:width
        if edges(i, j) == 1
            for theta = thetas
                rc = round(i + sind(theta) * radius);
                cc = round(j + cosd(theta) * radius);
                if not(any([rc < 1, cc < 1, rc > height, cc > width]))
                    accumulator(idivide(rc, bin_size, "ceil"), idivide(cc, bin_size, "ceil")) = ...
                        accumulator(idivide(rc, bin_size, "ceil"), idivide(cc, bin_size, "ceil")) + 1;
                end
            end
        end
    end
end

% Calculated accumulator matrix and normalize to display
% figure;
% acc_normal = mat2gray(accumulator);
% imshow(acc_normal);

% Get local maxima
localCenters = imregionalmax(accumulator);
accumulator = accumulator .* localCenters;

% Get points above threshold
voteThreshold = 0.8 * max(accumulator, [], "all");
constThreshold = 120;
threshold = max(voteThreshold, constThreshold);
fprintf("Threshold is: " + threshold + "\n");
[centerX, centerY] = ind2sub(size(accumulator), find(accumulator >= threshold));
centerY = centerY * double(bin_size);
centerX = centerX * double(bin_size);

% Swapped to match x and y-axis of viscircles()
centers = [centerY, centerX];
end