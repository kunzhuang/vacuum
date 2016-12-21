img=load('depth_canny_pt1.mat');

% Set bilateral filter parameters.
w     = 5;       % bilateral filter half-width
sigma = [3 0.1]; % bilateral filter standard deviations
img_i=double(img.depth)./1024;
% Apply bilateral filter to each image.
bflt_img = bfilter2(img_i,w,sigma);

depth=bflt_img*1024;


figure
imagesc(depth)
% 
% 
% 
cameracalib=load('D:\Dropbox\Thesis\code\TestCode\V1\calibrationSessionLR.mat');
[leftimg_col, rightimg_col] = rectifyStereoImages(imread('l.png'),imread('r.png'),cameracalib.calibrationSession.CameraParameters);
left_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters1;
right_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters2;
xyzPoints = reconstructScene(depth,cameracalib.calibrationSession.CameraParameters);
points3D = xyzPoints ./ 1000;
ptCloud = pointCloud(points3D,'Color',leftimg_col);
% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
% Visualize the point cloud
view(player3D, ptCloud);