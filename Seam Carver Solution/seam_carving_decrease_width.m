clear all
clc
close all

% im=imread('Input/Prague.jpg');
im=imread('Input/ut.jpg');

energyImg = energy_img(im);
imshow(energyImg);

nim=im;
nen=energyImg;
for i=1:100
    i;
    [nim,nen] = decrease_width(nim,nen);
end

imwrite(nim,'outputReduceWidthPrague.png');

figure;subplot(2,1,1);imshow(im);
title('Original Prague');
subplot(2,1,2);imshow(nim);
title('100 width decreased Prague');

figure;imshow(im);
title('Original Prague');
figure;imshow(nim);
title('100 width decreased Prague');
