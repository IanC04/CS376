function [allInliersCounts] = RANSACTracker(im, radius)
    img = im;

    % Get image of edge using edge-detection algorithm
    grayImg = im2gray(img);
    edges = edge(grayImg, "canny");

    inliersPer = [];

    N = inf;
    numIterations = 0; % Number of RANSAC iterations
    radiusThreshold = 5; % Threshold to consider a radius as a possible circle
    inlierThreshold = 2; % Threshold to consider a point as an inlier
    
    [rowIndices, colIndices] = find(edges == 1);
    % Ensure there are at least 3 non-zero elements in the matrix
    if any([numel(rowIndices) < 3 numel(colIndices) < 3])
        % Handle the case where there are fewer than 3 non-zero elements
        disp('There are fewer than 3 non-zero elements in the matrix.');
        return;
    end
    % Convert to (x, y) points
    edgePoints = horzcat(colIndices, rowIndices);

    while numIterations < N
        % Randomly select 3 unique indices
        randomIndices = randperm(numel(rowIndices), 3);

        % Get the corresponding row and column values for the selected
        % indices in the form p_i = (x, y)
        p1 = edgePoints(randomIndices(1, 1), :);
        p2 = edgePoints(randomIndices(1, 2), :);
        p3 = edgePoints(randomIndices(1, 3), :);
        
        % Calculate the circle parameters (center and radius) from the 3 points
        [circleCenter, circleRadius] = fitCircle(p1(1), p1(2), p2(1), p2(2), p3(1), p3(2));
        
        % Skip if the radius is significantly different from the expected radius
        if abs(circleRadius - radius) > radiusThreshold
            continue;
        end
        
        % Initialize inliers for this iteration
        inliers = 0;
        
        % Check each pixel in the image
        for i = 1:numel(rowIndices)
            point = edgePoints(i, :);
            x = point(1);
            y = point(2);

            % Calculate the distance between the current point and the circle
            distance = sqrt((x - circleCenter(1))^2 + (y - circleCenter(2))^2);

            % Check if the distance is within the inlier threshold
            if abs(distance - circleRadius) < inlierThreshold
                inliers = inliers + 1;
            end
        end
        
        % Update the best parameters if this iteration has more inliers
        inliersPer(end + 1, :) = inliers;

        p = 0.99;
        w = inliers / size(edgePoints, 1);
        n = 3;
        N = log(1 - p) / log(1 - w^n);
        numIterations = numIterations + 1;
    end
    allInliersCounts = inliersPer;
end

function [center, radius] = fitCircle(x1, y1, x2, y2, x3, y3)
    A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2;
    if A == 0
        center = [0 0];
        radius = inf;
        return;
    end
    B = (x1^2 + y1^2) * (y3 - y2) + (x2^2 + y2^2) * (y1 - y3) + (x3^2 + y3^2) * (y2 - y1);
    C = (x1^2 + y1^2) * (x2 - x3) + (x2^2 + y2^2) * (x3 - x1) + (x3^2 + y3^2) * (x1 - x2);
    D = (x1^2 + y1^2) * (x3 * y2 - x2 * y3) + (x2^2 + y2^2) * (x1 * y3 - x3 * y1) + (x3^2 + y3^2) * (x2 * y1 - x1 * y2);

    xc = -B / (2 * A);
    yc = -C / (2 * A);
    center = [xc yc];
    radius = sqrt((B^2 + C^2- 4 * A * D)/(4 * A^2));
end