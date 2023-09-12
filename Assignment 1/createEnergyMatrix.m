function energyImg = createEnergyMatrix(im)
grayImg=double(im2gray(im));

dx=grayImg(:,2:end)-grayImg(:,1:end-1);
dy=grayImg(2:end,:)-grayImg(1:end-1,:);

energyImg= sqrt(dx(1:end-1,:).^2+dy(:,1:end-1).^2);
energyImg(end+1,end+1)=0;
end