clc;clear;close all;

cameracalib=load('D:\Dropbox\Thesis\code\TestCode\V1\calibrationSessionLR.mat');

left_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters1;
right_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters2;

%leftimg=undistortImage(rgb2gray(imread('l.png')),left_cali);
%rightimg=undistortImage(rgb2gray(imread('r.png')),right_cali);
leftimg=rgb2gray(imread('l.png'));
rightimg=rgb2gray(imread('r.png'));



[left_rectf_crop, right_rectf_crop] = rectifyStereoImages(leftimg,rightimg,cameracalib.calibrationSession.CameraParameters);
[leftimg_col, rightimg_col] = rectifyStereoImages(imread('l.png'),imread('r.png'),cameracalib.calibrationSession.CameraParameters);



% left_rectf_crop=imcrop(left_rectf_crop,[300 150 670 300]);
% right_rectf_crop=imcrop(right_rectf_crop,[300 150 670 300]);
% 
% 
% 
% figure
% imshow(left_rectf_crop)
% figure
% imshow(right_rectf_crop)
% 
% 
% leftimg_col=imcrop(leftimg_col,[300 150 670 300]);
% rightimg_col=imcrop(rightimg_col,[300 150 670 300]);


% [Gmagl, left_rectf_crop] = imgradient(left_rectf_crop,'prewitt');
% [Gmagr, right_rectf_crop] = imgradient(right_rectf_crop,'prewitt');

% left_rectf_crop=imgaussfilt(left_rectf_crop, 1);
% right_rectf_crop=imgaussfilt(right_rectf_crop, 1);

left_rectf_crop1=edge(left_rectf_crop,'Sobel',0);
right_rectf_crop1=edge(right_rectf_crop,'Sobel',0);

left_rectf_crop(left_rectf_crop1)=1;
right_rectf_crop(right_rectf_crop1)=1;

disparityRange = [64 1024];
% 0.26 to 4.1 meters


im_l=left_rectf_crop;
im_r=right_rectf_crop;

%Initialize variables
w=30;
dispmax=256;

%Resize images
im_l_res=zeros(size(im_l,1)+2*w,size(im_l,2)+2*w+dispmax);
im_r_res=zeros(size(im_r,1)+2*w,size(im_r,2)+2*w+dispmax);
 
im_l_res(w:end-1-w,w:end-1-w-dispmax)=im_l;
im_r_res(w:end-1-w,w:end-1-w-dispmax)=im_r;


for y=w+1:1:size(im_l_res,1)-w %For each epipolar line (row)
    y %to see where it is
    for x=w+1:1:size(im_l_res,2)-w-dispmax  %For each pixel on that row
       
        left=im_l_res(y-w:y+w,x-w:x+w);
        
            parfor disp=0:1:dispmax
                right=im_r_res(y-w:y+w,x-w+disp:x+w+disp);
                SAD(disp+1)=sum(abs(left(:)-right(:))); %Take the sum of absolute difference
            end
            [temp,depth(y-w,x-w)]=min(SAD);
    end
end

% [SAD_min,SAD_min_loc]=min(SAD,[],3);
% imagesc(SAD_min_loc)
% 
% 
% 
xyzPoints = reconstructScene(depth,cameracalib.calibrationSession.CameraParameters);
points3D = xyzPoints ./ 1000;
ptCloud = pointCloud(points3D,'Color',leftimg_col);
% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
% Visualize the point cloud
view(player3D, ptCloud);