clc, clearvars, close all;

ut_image = imread("Assignment1_pics\ut.jpg");
river_image = imread("Assignment1_pics\river.jpg");
gsw_image = imread("Assignment1_pics\gsw.jpg");
memory_image = imread("Assignment1_pics\persistence_of_memory.jpg");
toucan_image = imread("Assignment1_pics\toucan.jpg");

%{
% Question 1
figure("Name", "Seam removal pictures")

subplot(2, 3, 1);
imshow(ut_image);
title("Original")

subplot(2, 3, 2);
imshow(removeVertical(ut_image, 100));
title("Removed 100 Vertical Seams");

subplot(2, 3, 3);
imshow(removeHorizontal(ut_image, 100));
title("Removed 100 Horizontal Seams");

subplot(2, 3, 4);
imshow(river_image);
title("Original")

subplot(2, 3, 5);
imshow(removeVertical(river_image, 100));
title("Removed 100 Vertical Seams");

subplot(2, 3, 6);
imshow(removeHorizontal(river_image, 100));
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
%}

%{
% Question 5-P1
figure("Name", "Custom results");

subplot(3, 4, 1);
imshow(gsw_image);
title("Original(600×800×3)")

subplot(3, 4, 2);
picture = gsw_image;
picture = removeHorizontal(picture, 200);
imshow(picture);
title("200 Horizontal Seams Resizing(400×800×3)");

subplot(3, 4, 3);
picture = gsw_image;
[r, c] = size(picture, [1  2]);
picture = imresize(picture, [r-200 c], "nearest");
imshow(picture);
title("Nearest Neighbor Resizing(400×800×3)");

subplot(3, 4, 4);
picture = gsw_image;
[r, c] = size(picture, [1  2]);
picture = imresize(picture, [r-200 c], "bicubic");
imshow(picture);
title("Bicubic Interpolation Resizing(400×800×3)");

subplot(3, 4, 5);
imshow(memory_image);
title("Original(221×300×3)")

subplot(3, 4, 6);
picture = memory_image;
picture = removeVertical(picture, 100);
imshow(picture);
title("100 Vertical Seams Resizing(221×200×3)");

subplot(3, 4, 7);
picture = memory_image;
[r, c] = size(picture, [1  2]);
picture = imresize(picture, [r c-100], "nearest");
imshow(picture);
title("Nearest Neighbor Resizing(221×200×3)");

subplot(3, 4, 8);
picture = memory_image;
[r, c] = size(picture, [1  2]);
picture = imresize(picture, [r c-100], "bicubic");
imshow(picture);
title("Bicubic Interpolation Resizing(221×200×3)");

subplot(3, 4, 9);
imshow(toucan_image);
title("Original(419×640×3)")

subplot(3, 4, 10);
picture = toucan_image;
picture = removeVertical(picture, 100);
imshow(picture);
title("100 Vertical Seams Resizing(419×540×3)");

subplot(3, 4, 11);
picture = toucan_image;
[r, c] = size(picture, [1  2]);
picture = imresize(picture, [r c-100], "nearest");
imshow(picture);
title("Nearest Neighbor Resizing(419×540×3)");

subplot(3, 4, 12);
picture = toucan_image;
[r, c] = size(picture, [1  2]);
picture = imresize(picture, [r c-100], "bicubic");
imshow(picture);
title("Bicubic Interpolation Resizing(419×540×3)");
%}

%{
% Question 5-P2 All Seams Highlighted

figure("Name","All Vertical Removed Seams");

picture = gsw_image;
[highlighted, image] = highlightNumberOfSeams(picture, 200, "Vertical");
subplot(3,3,1);
imshow(picture);
title("Original");
subplot(3,3,2);
imshow(highlighted);
title("Highlighted 200 Seams");
subplot(3,3,3);
imshow(image);
title("Resulting Image");

picture = memory_image;
[highlighted, image] = highlightNumberOfSeams(picture, 100, "Vertical");
subplot(3,3,4);
imshow(picture);
title("Original");
subplot(3,3,5);
imshow(highlighted);
title("Highlighted 100 Seams");
subplot(3,3,6);
imshow(image);
title("Resulting Image");

picture = toucan_image;
[highlighted, image] = highlightNumberOfSeams(picture, 100, "Vertical");
subplot(3,3,7);
imshow(picture);
title("Original");
subplot(3,3,8);
imshow(highlighted);
title("Highlighted 100 Seams");
subplot(3,3,9);
imshow(image);
title("Resulting Image");
%}

%{
% Question 5-P3 All Seams Highlighted

figure("Name","All Vertical Removed Seams");

picture = gsw_image;
[highlighted, image] = highlightNumberOfSeams(picture, 200, "Horizontal");
subplot(3,3,1);
imshow(picture);
title("Original");
subplot(3,3,2);
imshow(highlighted);
title("Highlighted 200 Seams");
subplot(3,3,3);
imshow(image);
title("Resulting Image");

picture = memory_image;
[highlighted, image] = highlightNumberOfSeams(picture, 100, "Horizontal");
subplot(3,3,4);
imshow(picture);
title("Original");
subplot(3,3,5);
imshow(highlighted);
title("Highlighted 100 Seams");
subplot(3,3,6);
imshow(image);
title("Resulting Image");

picture = toucan_image;
[highlighted, image] = highlightNumberOfSeams(picture, 100, "Horizontal");
subplot(3,3,7);
imshow(picture);
title("Original");
subplot(3,3,8);
imshow(highlighted);
title("Highlighted 100 Seams");
subplot(3,3,9);
imshow(image);
title("Resulting Image");
%}