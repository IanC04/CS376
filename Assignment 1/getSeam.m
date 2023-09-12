function output = getSeam(energyImg, seamDirection)
cumulativeEnergyMatrix = createMinimumEnergyMatrix(energyImg, seamDirection);

if strcmp(seamDirection, "Vertical")
    verticalSeam=zeros(size(cumulativeEnergyMatrix,1),1);

    [~, b]=min(cumulativeEnergyMatrix(end,:));
    verticalSeam(end)=b;
    for i=size(verticalSeam,1)-1:-1:1
        [~, bb]=min(cumulativeEnergyMatrix(i,b-1:b+1));
        b=b+bb-2;
        verticalSeam(i)=b;
    end
    output = verticalSeam;
end

if strcmp(seamDirection, "Horizontal")
    horizontalSeam=zeros(size(cumulativeEnergyMatrix,2),1);

    [~, b]=min(cumulativeEnergyMatrix(:,end));
    horizontalSeam(end)=b;
    for i=size(horizontalSeam,1)-1:-1:1
        [~, bb]=min(cumulativeEnergyMatrix(b-1:b+1,i));
        b=b+bb-2;
        horizontalSeam(i)=b;
    end
    output = horizontalSeam;
end
end