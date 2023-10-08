function [centers] = detectCirclesHT(im, radius)
    img = im;
    thetas = 0:359;
    [height, width] = size(img, [1 2]);

    % Get image of edge using edge-detection algorithm
    grayImg = im2gray(img);
    edges = edge(grayImg, "canny");

    % Create accumulator array and use polar coordinates
    accumulator = zeros(height, width);

    % Calculate Hough radii
    for i = 1:height
        for j = 1:width
            if edges(i, j) == 1
                for theta = thetas
                    rc = round(i + sind(theta) * radius);
                    cc = round(j + cosd(theta) * radius);
                    if not(any([rc < 1, cc < 1, rc > height, cc > width]))
                        accumulator(rc, cc) = accumulator(rc, cc) + 1;
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
    threshold = 120;
    % threshold = min(max(accumulator,[],"all"), 120);
    [centerX, centerY] = ind2sub(size(accumulator), find(accumulator >= threshold));
    
    % Swapped to match x and y-axis of viscircles()
    centers = [centerY, centerX];
end