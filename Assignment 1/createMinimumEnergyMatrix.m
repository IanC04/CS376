function [output] = createMinimumEnergyMatrix(energyImg, direction)
    output = zeros(size(energyImg));    
    if strcmp(direction, "Vertical")
        output(1,:) = energyImg(1,:);
        output(2:end,1) = Inf;
        output(2:end,end) = Inf;

    for i=2:size(output,1)
        row = movmin(output(i-1,:),3);
        for j=2:size(output,2)-1
            output(i,j)=energyImg(i,j) + row(j);
        end
    end
    end

    if strcmp(direction, "Horizontal")
        output(:,1) = energyImg(:,1);
        output(1,2:end) = Inf;
        output(end,2:end) = Inf;

    for i=2:size(output,2)
        col = movmin(output(:,i-1),3);
        for j=2:size(output,1)-1
            output(j,i)=energyImg(j,i) + col(j);
        end
    end
    end
end