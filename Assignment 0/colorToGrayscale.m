% map a color image into a grayscale image.
function picture = colorToGrayscale(file)
picture = imread(file);
picture = im2gray(picture);
end