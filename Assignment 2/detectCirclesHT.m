function [centers] = detectCirclesHT(im, radius)
    img = im;
    thetas = 1:360;
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
                    x_circ = round(cosd(theta) * radius + x);
                    y_circ = round(sind(theta) * radius + y);
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

    % Get points above threshold
    threshold = min(max(accumulator,[],"all"), 180);
    [centerX, centerY] = ind2sub(size(accumulator), find(accumulator >= threshold));
    centers = [centerX, centerY];
end