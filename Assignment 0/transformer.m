picture = "toucan.jpg";
subplot(6,2,1);
imshow(picture);
title("Original");
subplot(6,2,2);
imshow(colorToGrayscale(picture));
title("Grayscale");

subplot(6,2,3);
imshow(colorToGrayscale(picture));
title("Grayscale");
subplot(6,2,4);
imshow(grayToNegative(picture));
title("Negative");

subplot(6,2,5);
imshow(picture);
title("Original");
subplot(6,2,6);
imshow(flip(picture));
title("Flipped");

subplot(6,2,7);
imshow(picture);
title("Original");
subplot(6,2,8);
imshow(rgPlusReduce(picture));
title("Channels Flipped then reduced");

subplot(6,2,9);
imshow(colorToGrayscale(picture));
title("Grayscale");
subplot(6,2,10);
imshow(grayscaleRandomized(picture));
title("Grayscale Randomized");

subplot(6,2,11);
imshow(colorToGrayscale(picture));
title("Grayscale");
subplot(6,2,12);
imshow(plotConditional(picture));
title("Grayscale Conditional");