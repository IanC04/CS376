function [labelIm] = clusterPixels(Im, k)
    img = im2double(Im);

    totalIterations = 0;

    % Use k number of points as means of clusters
    [rows, columns] = size(img, [1 2]);
    rowIndices = randperm(size(img, 1), k);
    columnIndices = randperm(size(img, 2), k);

    % Initial mean values
    means = zeros(k, 3);
    for i = 1:k
        means(i,:) = img(rowIndices(i), columnIndices(i),:);
    end

    % Cluster indices
    clusterIndices = randi(k, size(img, [1 2]));


    % Save current mean values to compare to global, have -1 because
    % impossible value
    curMeans = means;

    equal = false;

    while (~equal)
        means = curMeans;
        for i = 1:rows
            for j = 1:columns
                allDistances = distanceToMeans(curMeans, img(i, j, :));
                [~, index] = min(allDistances);
                clusterIndices(i, j) = index;
            end
        end
        curMeans = getMeans(img, clusterIndices, k);

        totalIterations = totalIterations + 1;
        fprintf("Current iteration: " + totalIterations + "\n");

        equal = isequal(means, curMeans);
    end

    labelIm = clusterIndices;
end

function [means] = getMeans(img, clusterMatrix, k)
    means = zeros(k, 4);
    [rows, columns] = size(clusterMatrix, [1 2]);

    for i = 1:rows
        for j = 1:columns
            index = clusterMatrix(i, j);
            new = [reshape(img(i, j,:), [1 3]), 1.0];
            original = means(index,:);
            for col = 1:4
                means(index,col) = original(col) + new(col);
            end
        end
    end

    for i = 1:k
        means(i,:) = means(i,:) / means(i, 4);
    end

    means = means(:,1:end - 1);
end

function [distances] = distanceToMeans(allMeans, color)
    % Get the point with value color's distance to each cluster in allMeans
    distances = zeros(1, size(allMeans, 1));
    color = reshape(color, [1 3]);

    for i = 1:numel(distances)
        distances(1, i) = getDistance(allMeans(i,:), color);
    end
end

function [distance] = getDistance(color1, color2)
    r1 = double(color1(1, 1));
    g1 = double(color1(1, 2));
    b1 = double(color1(1, 3));

    r2 = double(color2(1, 1));
    g2 = double(color2(1, 2));
    b2 = double(color2(1, 3));

    result = sqrt((r2 - r1)^2 + (g2 - g1)^2 + (b2 - b1)^2);
    distance = result;
end