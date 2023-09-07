function picture = rgPlusReduce(file)
picture = imread(file);
picture(:,:,1:2) = flipdim(picture(:,:,1:2), 3);
picture(:,:,1) = picture(:,:,1) / 2;