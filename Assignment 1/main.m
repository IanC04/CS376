% Close all previous variables and windows
clc, clearvars, close all;

figure("Name", "Images and their Carvings");

% Display original
original = imread("Assignment1_pics\ut.jpg");
subplot(3, 3, 1);
imshow(original);
title("Original");

% Display energy matrix
subplot(3, 3, 2);
energyMatrix = createEnergyMatrix(original);
maxIntensity = max(max(energyMatrix));
imshow(energyMatrix./maxIntensity);
title("Energy Matrix(Scaled)");

% Display the cumulative minimum energy matrix
subplot(3, 3, 3);
minimumEnergyMatrix = createMinimumEnergyMatrix(energyMatrix);
maxIntensity = max(max(minimumEnergyMatrix(minimumEnergyMatrix < Inf)));
imshow(minimumEnergyMatrix./maxIntensity);
title("Minimum Energy Matrix(Scaled)");

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