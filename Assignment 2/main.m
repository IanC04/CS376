clearvars, close all;

coinsImg = imread("Assignment2_pics\coins.jpg");
planetsImg = imread("Assignment2_pics\planets.jpg");
centers = detectCirclesHT(coinsImg, 100);
% detectCirclesHT(planetsImg, 1);
% detectCirclesRANSAC(coinsImg, 1);
% detectCirclesRANSAC(planetsImg, 1);