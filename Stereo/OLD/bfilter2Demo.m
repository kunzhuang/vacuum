
% Note: Must be double precision in the interval [0,1].
img = depth/max(max(depth));



% Set bilateral filter parameters.
w     = 5;       % bilateral filter half-width
sigma = [2 0.1]; % bilateral filter standard deviations

%5 3 0.1
% Apply bilateral filter to each image.
bflt_img = bfilter2(img,w,sigma)*max(max(depth));
plot(bflt_img(30,:))
hold on
plot(depth(30,:))

figure
imagesc(depth)
figure
imagesc(bflt_img)