function picture = flip(file)
picture = imread(file);
picture = fliplr(picture);
end