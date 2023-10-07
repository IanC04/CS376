function [highlighted] = highlightCircles(image, centers, radius)
    thetas = 1:360;
    highlighted = image;
    [r, c] = size(image, [1 2]);
    redPatch = [255 255 255; 255 255 255; 255 255 255];
    redPatch(:,:,2) = [0 0 0; 0 0 0; 0 0 0];
    redPatch(:,:,3) = [0 0 0; 0 0 0; 0 0 0];

    for i = 1:size(centers,1)
        x = centers(i, 1) - 1;
        y = centers(i, 2) - 1;
        for theta = thetas
            x_circ = round(sin(theta) * radius + x);
            y_circ = round(cos(theta) * radius + y);
            if not(any([x_circ - 1 < 1, y_circ - 1 < 1, x_circ + 1 > r, y_circ + 1 > c]))
                highlighted(x_circ - 1 : x_circ + 1, y_circ - 1 : y_circ + 1, :) = redPatch;
            end
        end
    end
end