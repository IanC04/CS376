function [highlightedImg, result] = highlightNumberOfSeams(im, numPixels, direction)
picture = im;
output = im;
seam_positions = false(size(output, [1 2]));
for i=1:numPixels
    seam = getSeam(createEnergyMatrix(picture), direction);
    picture = removeSeams(picture, 1, direction);
    for j=1:length(seam)
        if strcmp(direction, "Vertical")
        seam(j) = seam(j) + sum(seam_positions(j, 1:seam(j)) > 0);
        seam_positions(j, seam(j)) = 1;
        end
        
        if strcmp(direction, "Horizontal")
        seam(j) = seam(j) + sum(seam_positions(1:seam(j), j) > 0);
        seam_positions(seam(j), j) = 1;
        end
    end
    output = highlightSeam(output, seam, direction);
end
result = picture;
highlightedImg = output;
end