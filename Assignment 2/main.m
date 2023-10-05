coinsImg = imread("Assignment2_pics\coins.jpg");
planetsImg = imread("Assignment2_pics\planets.jpg");
detectCirclesHT(coinsImg, 1);
detectCirclesRANSAC(planetsImg, 1);