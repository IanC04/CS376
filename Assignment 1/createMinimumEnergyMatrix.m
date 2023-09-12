function [output] = createMinimumEnergyMatrix(energyImg)
    output = zeros(size(energyImg));
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