clearvars, close all;

coinsImg = imread("Assignment2_pics\coins.jpg");
planetsImg = imread("Assignment2_pics\planets.jpg");
gumballsImg = imread("Assignment2_pics\gumballs.jpg");
snakeImg = imread("Assignment2_pics\snake.jpg");
twinsImg = imread("Assignment2_pics\twins.jpg");
toucanImg = imread("Assignment2_pics\toucan.jpg");
gswImg = imread("Assignment2_pics\gsw.jpg");

% showAccumulator(coinsImg, radius);

% tiledlayout(3, 3, "Padding","tight", "TileSpacing","compact");
% 
% nexttile;
% imshow(coinsImg);
% title("Coins");
% 
% nexttile;
% imshow(planetsImg);
% title("Planets");
% 
% nexttile;
% imshow(gswImg);
% title("GSW");
% 
% radius = 100;
% binsize = 1;
% nexttile;
% centers = detectCirclesHT(coinsImg, radius);
% imshow(coinsImg);
% hold on;
% viscircles(centers, radius);
% title("Hough-Circle in Coins of radius " + radius);
% 
% radius = 110;
% nexttile;
% centers = detectCirclesHT(planetsImg, radius);
% imshow(planetsImg);
% hold on;
% viscircles(centers, radius);
% title("Hough-Circle in Planets of radius " + radius);
% 
% radius = 20;
% nexttile;
% centers = detectCirclesHT(gswImg, radius);
% imshow(gswImg);
% hold on;
% viscircles(centers, radius);
% title("Hough-Circle in GSW of radius " + radius);
% 
% radius = 100;
% nexttile;
% centers = detectCirclesRANSAC(coinsImg, radius);
% imshow(coinsImg);
% hold on;
% viscircles(centers, radius);
% title("RANSAC-Circle in Coins of radius " + radius);
% 
% radius = 110;
% nexttile;
% centers = detectCirclesRANSAC(planetsImg, radius);
% imshow(planetsImg);
% hold on;
% viscircles(centers, radius);
% title("RANSAC-Circle in Planets of radius " + radius);
% 
% radius = 20;
% nexttile;
% centers = detectCirclesRANSAC(gswImg, radius);
% imshow(gswImg);
% hold on;
% viscircles(centers, radius);
% title("RANSAC-Circle in GSW of radius " + radius);

% Hough Transform

% tiledlayout(1, 2, "Padding","tight", "TileSpacing","compact");
%
% radius = 100;
% binsize = 1;
% nexttile;
% centers = detectCirclesHT(coinsImg, radius);
% imshow(coinsImg);
% hold on;
% viscircles(centers, radius);
% title("Hough-Circle in Coins of radius " + radius + " with bin size: " + binsize);
%
% radius = 105;
% nexttile;
% centers = detectCirclesHT(planetsImg, radius);
% imshow(planetsImg);
% hold on;
% viscircles(centers, radius);
% title("Hough-Circle in Planets of radius " + radius + " with bin size: " + binsize);

% RANSAC
%
% tiledlayout(1, 2, "Padding","tight", "TileSpacing","compact");
%
% radius = 100;
% nexttile;
% centers = detectCirclesRANSAC(coinsImg, radius);
% imshow(coinsImg);
% hold on;
% viscircles(centers, radius);
% title("RANSAC-Circle in Coins of radius " + radius);
%
% radius = 110;
% nexttile;
% centers = detectCirclesRANSAC(planetsImg, radius);
% imshow(planetsImg);
% hold on;
% viscircles(centers, radius);
% title("RANSAC-Circle in Planets of radius " + radius);
%

% RANSAC Progression

tiledlayout(1, 3, "Padding","tight", "TileSpacing","compact");

nexttile;
radius = 40;
[allInliers, bestInliers, centers] = RANSACTracker(coinsImg, radius);
plot(bestInliers);
title("Progression of RANSAC algorithm");
xlabel("Iterations");
ylabel("Best Number of Inliers for a Circle of radius: " + radius);

nexttile;
plot(allInliers);
title("Progression of RANSAC algorithm");
xlabel("Iterations");
ylabel("Total Number of Inliers for a Circle of radius: " + radius);

nexttile;
plot(allInliers);
title("Detected Circles of radius: " + radius);
imshow(coinsImg);
hold on;
viscircles(centers, radius);

% K-Means Clustering

% % Max clusters
% k = 10;
% % Images are gumballsImg, snakeImg twinsImg
% % image = gumballsImg;
% % image = snakeImg;
% % image = twinsImg;
% image = toucanImg;
% 
% tiledlayout(4, 2, "Padding","tight", "TileSpacing","compact");
% 
% nexttile;
% imshow(image, "InitialMagnification", "fit", "Border","tight");
% title("Original image");
% 
% nexttile;
% grayImg = im2gray(image);
% edges = edge(grayImg, "canny");
% imshow(edges, "InitialMagnification", "fit", "Border","tight");
% title("Edges of original image");
% 
% for i = 2:4:k
%     nexttile;
%     labelImg = clusterPixels(image, i);
%     imshow(mat2gray(labelImg), "InitialMagnification", "fit", "Border","tight");
%     title("Image using " + i + " clusters");
% 
%     nexttile;
%     boundaryImg = boundaryPixels(labelImg);
%     imshow(mat2gray(boundaryImg), "InitialMagnification", "fit", "Border","tight");
%     title("Edges of the " + i + " clusters");
% end