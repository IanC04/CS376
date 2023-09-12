ut_image = imread("Assignment1_pics\ut.jpg");
river_image = imread("Assignment1_pics\river.jpg");
memory_image = imread("Assignment1_pics\persistence_of_memory.jpg");
toucan_image = imread("Assignment1_pics\toucan.jpg");

%{
% Question 1
figure("Name", "Seam removal pictures")

subplot(1, 3, 1);
imshow(ut_image);
title("Original")

subplot(1, 3, 2);
imshow(removeVertical(ut_image, 100));
title("Removed 100 Vertical Seams");

subplot(1, 3, 3);
imshow(removeHorizontal(ut_image, 100));
title("Removed 100 Horizontal Seams");
%}

%{
% Question 2
figure("Name", "Energy function pictures")

subplot(1, 4, 1);
imshow(ut_image);
title("Original")

subplot(1, 4, 2);
imshow(createEnergyMatrix(ut_image));
title("Energy Function");

subplot(1, 4, 3);
imagesc(createMinimumEnergyMatrix(createEnergyMatrix(ut_image), "Vertical"));
title("Cumulative Minimum Energy Function-Vertical");

subplot(1, 4, 4);
imagesc(createMinimumEnergyMatrix(createEnergyMatrix(ut_image), "Horizontal"));
title("Cumulative Minimum Energy Function-Horizontal");

% The reason for these outputs is that on top of each object there's a
% triangle pointing to it, since for a object or background, the gradient
% is small, but, when edges to objects are detected through large
% gradients, the energy path will prioritize other paths which don't pass
% through manny edges.
%}

%{
% Question 3
figure("Name", "Seam pictures")

subplot(1, 3, 1);
imshow(ut_image);
title("Original")

subplot(1, 3, 2);
seam = getSeam(createEnergyMatrix(ut_image), "Vertical");
imshow(highlightSeam(ut_image, seam, "Vertical"));
title("Vertical Seam");

subplot(1, 3, 3);
seam = getSeam(createEnergyMatrix(ut_image), "Horizontal");
imshow(highlightSeam(ut_image, seam, "Horizontal"));
title("Horizontal Seam");

% These are the optimal seams since they include the blue sky background,
% which is mostly the same color. Therefore, their cumulative gradient will
% be small. However, in the vertical gradient, the trees in the foreground
% cover the entire bottom half, so the path passed through the patch which
% are mostly the same shade of the tree leaves.
%}

%{
% Question 4
figure("Name", "Altered energy function resulting pictures")

subplot(1, 3, 1);
imshow(ut_image);
title("Original")

subplot(1, 3, 2);
seam = getSeam(createEnergyMatrix(ut_image), "Vertical");
imshow(highlightSeam(imgaussfilt(ut_image, 5), seam, "Vertical"));
title("Blurred Image of Vertical Seam");

ut_image = imgaussfilt(ut_image, 5);

subplot(1, 3, 3);
seam = getSeam(createEnergyMatrix(ut_image), "Vertical");
imshow(highlightSeam(ut_image, seam, "Vertical"));
title("Vertical Seam of Blurred Image");

% The results prove that blurring images first then calculating seams may
% produce different results then calculating seams then blurring the image.
% This is because the gradients of each pixel now are smaller since its
% value is closer to its neighbors.
%}

% Question 5
figure("Name", "Custom results")

subplot(3, 4, 1);
imshow(river_image);
title("Original")

subplot(3, 4, 2);
picture = river_image;
picture = removeVertical(picture, 100);
picture = removeHorizontal(picture, 100);
imshow(picture);
title("Seam Resizing of Image");

subplot(3, 4, 3);
picture = river_image;
[r, c] = size(picture, [1  2]);
picture = imresize(picture, [r c] - 100);
imshow(picture);
title("MATLAB Resizing of Image");

% Original: 318×424×3   Resized: 218×324×3 uint8