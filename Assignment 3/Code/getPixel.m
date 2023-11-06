close all;

source = imread("../Assignment 3 Pics/SourceImage.jpg");
target = imread("../Assignment 3 Pics/TargetImage.jpg");

figure;
imshow(source);
zoom on;
waitfor(gcf, 'CurrentCharacter', char(13));
[x1,y1] = ginput(8);
x1 = round(x1);
y1 = round(y1);
axis on;
hold on;
plot(x1,y1, 'ro', 'MarkerSize', 5);
figure;
imshow(target);
zoom on;
waitfor(gcf, 'CurrentCharacter', char(13));
[x2,y2] = ginput(8);
x2 = round(x2);
y2 = round(y2);
axis on;
hold on;
plot(x2,y2, 'ro', 'MarkerSize', 5);