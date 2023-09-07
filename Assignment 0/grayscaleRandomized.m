function picture = grayscaleRandomized(file)
picture = colorToGrayscale(file);
[M,N] = size(picture);
picture = picture + uint8(randi([0,510],M,N) - 255);
picture(picture < 0) = 0;
picture(picture > 255) = 255;
end