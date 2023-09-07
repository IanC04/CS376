function picture = grayToNegative(file)
picture = colorToGrayscale(file);
picture = 255 - picture;
end