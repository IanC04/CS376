% Close all previous variables and windows
clc, clearvars, close all;

% Display original
original = imread("Assignment1_pics\ut.jpg");
subplot(3, 3, 1);
imshow(original);
title("Original");

% Display energy matrix
subplot(3, 3, 2);
energyMatrix = createEnergyMatrix(original);
imshow(energyMatrix);
title("Energy Matrix");

% Display the cumulative minimum energy matrix
subplot(3, 3, 3);
minimumEnergyMatrix = createMinimumEnergyMatrix(energyMatrix);
imshow(minimumEnergyMatrix);
title("Minimum Energy Matrix");

% Highlight the seam
subplot(3, 3, 4);
imshow(original);
hold on;
verticalSeam = getSeam(minimumEnergyMatrix, "Vertical");
seamedImg = highlightSeam(original, verticalSeam, "Vertical");
imshow(seamedImg);
title("Calculated Vertical Seam");

% Remove chosen amount of seams
subplot(3, 3, 5);
removedSeams = removeSeams(original, 100, "Vertical");
imshow(removedSeams);
title("Removed Seams");