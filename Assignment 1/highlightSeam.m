function output = highlightSeam(im,seam,seamDirection)
output = im;
if strcmp(seamDirection, "Vertical") 
    for i=1:length(seam)
    output(i,seam(i),:)=[255 0 0];
    end
end 
if strcmp(seamDirection, "Horizontal")
    for j=1:length(seam)
    output(seam(j), j,:)=[255 0 0];
    end
end
end