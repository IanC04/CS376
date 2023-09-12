function [output] = removeSeams(im, numPixels, direction)
output = im;

if strcmp(direction, "Vertical")
    for times = 1:numPixels
        energyImg = createEnergyMatrix(output);
        verticalSeam = getSeam(energyImg, direction);
    for i=1:length(verticalSeam)
    output(i,verticalSeam(i):end-1,:)=output(i,verticalSeam(i)+1:end,:);
    end
    output = output(:,1:end-1,:);
    end
end

if strcmp(direction, "Horizontal")
    for times = 1:numPixels
        energyImg = createEnergyMatrix(output);
        horizontalSeam = getSeam(energyImg, direction);
    for j=1:length(horizontalSeam)
    output(horizontalSeam(j):end-1,j,:)=output(horizontalSeam(j)+1:end,j,:);
    end
    output = output(1:end-1,:,:);
    end
end
end