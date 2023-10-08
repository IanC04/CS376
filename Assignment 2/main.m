clearvars, close all;

coinsImg = imread("Assignment2_pics\coins.jpg");
planetsImg = imread("Assignment2_pics\planets.jpg");
gumballsImg = imread("Assignment2_pics\gumballs.jpg");
snakeImg = imread("Assignment2_pics\snake.jpg");
twinsImg = imread("Assignment2_pics\twins.jpg");

% showAccumulator(coinsImg, radius);

% % Hough Transform
% radius = 100;
% subplot(2, 2, 1);
% centers = detectCirclesHT(coinsImg, radius);
% imshow(coinsImg);
% hold on;
% viscircles(centers, radius);
% title("Hough-Circle in Coins of radius " + radius);
% 
% radius = 110;
% subplot(2, 2, 2);
% centers = detectCirclesHT(planetsImg, radius);
% imshow(planetsImg);
% hold on;
% viscircles(centers, radius);
% title("Hough-Circle in Planets of radius " + radius);

% % RANSAC
% radius = 100;
% subplot(2, 2, 3);
% centers = detectCirclesRANSAC(coinsImg, radius);
% imshow(coinsImg);
% hold on;
% viscircles(centers, radius);
% title("RANSAC-Circle in Coins of radius " + radius);
% 
% radius = 110;
% subplot(2, 2, 4);
% centers = detectCirclesRANSAC(planetsImg, radius);
% imshow(planetsImg);
% hold on;
% viscircles(centers, radius);
% title("RANSAC-Circle in Planets of radius " + radius);

% radius = 100;
% inliers = RANSACTracker(coinsImg, radius);
% plot(inliers);
% title("Progression of RANSAC algorithm");
% xlabel("Iterations"); 
% ylabel("Number of Inliers");

% K-Means Clustering

k = 10;
% subplot(1, 2, 1);
% labelImg = clusterPixels(gumballsImg, k);
% imshow(mat2gray(labelImg));
% title("Image using only " + k + " clusters");
% 
% subplot(1, 2, 2);
% boundaryImg = boundaryPixels(labelImg);
% imshow(mat2gray(boundaryImg));
% title("Edges of the " + k + " clusters");

% subplot(1, 2, 1);
% labelImg = clusterPixels(snakeImg, k);
% imshow(mat2gray(labelImg));
% title("Image using only " + k + " clusters");
% 
% subplot(1, 2, 2);
% boundaryImg = boundaryPixels(labelImg);
% imshow(mat2gray(boundaryImg));
% title("Edges of the " + k + " clusters");

subplot(1, 2, 1);
labelImg = clusterPixels(twinsImg, k);
imshow(mat2gray(labelImg));
title("Image using only " + k + " clusters");

subplot(1, 2, 2);
boundaryImg = boundaryPixels(labelImg);
imshow(mat2gray(boundaryImg));
title("Edges of the " + k + " clusters");