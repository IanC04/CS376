function output = highlightSeam(im,seam,seamDirection)
output = im;
if strcmp("Vertical",seamDirection) 
    for i=1:length(seam)
    output(i,seam(i),:)=[255 0 0];
    end
end 
if strcmp("Horizontal",seamDirection)
    for j=1:length(seam)
    output(seam(j), j,:)=[255 0 0];
    end
end
end