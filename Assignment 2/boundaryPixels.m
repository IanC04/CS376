function [boundaryIm] = boundaryPixels(labelIm)
    img = labelIm;
    boundaryIm = edge(img, "canny");
end