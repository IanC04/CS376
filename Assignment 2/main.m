clearvars, close all;

radius = 25;
coinsImg = imread("Assignment2_pics\coins.jpg");
planetsImg = imread("Assignment2_pics\planets.jpg");
centers = detectCirclesHT(coinsImg, radius);
% highlightedImg = highlightCircles(coinsImg, centers, radius);
% imshow(highlightedImg);
% detectCirclesHT(planetsImg, radius);
detectCirclesRANSAC(coinsImg, radius);
highlightedImg = highlightCircles(coinsImg, centers, radius);
imshow(highlightedImg);
% detectCirclesRANSAC(planetsImg, radius);