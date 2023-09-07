function picture = plotConditional(file)
picture = colorToGrayscale(file);
picture = picture == 128;
end