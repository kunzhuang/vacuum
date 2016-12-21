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



left_rectf_crop=imcrop(left_rectf_crop,[300 150 670 300]);
right_rectf_crop=imcrop(right_rectf_crop,[300 150 670 300]);
leftimg_col=imcrop(leftimg_col,[300 150 670 300]);
rightimg_col=imcrop(rightimg_col,[300 150 670 300]);



% [Gmagl, left_rectf_crop] = imgradient(left_rectf_crop,'prewitt');
% [Gmagr, right_rectf_crop] = imgradient(right_rectf_crop,'prewitt');

%left_rectf_crop=imgaussfilt(left_rectf_crop, 1);
%right_rectf_crop=imgaussfilt(right_rectf_crop, 1);

%left_rectf_crop1=edge(left_rectf_crop,'Sobel',0);
%right_rectf_crop1=edge(right_rectf_crop,'Sobel',0);

%left_rectf_crop(left_rectf_crop1)=1;
%right_rectf_crop(right_rectf_crop1)=1;

disparityRange = [64 1024];
% 0.26 to 4.1 meters

% figure
% imshow(left_rectf_crop)
% figure
% imshow(right_rectf_crop)


left_in=left_rectf_crop;
right_in=right_rectf_crop;

%Initialize variables
w=25; %even numbers only
dispmax=256;
dispmin=64;
% mod(w,2)
w=w-1; %left+center+right
wd2=w/2;

%Resize images
%left_in=[1 2 3 4 5 6 7 8;1 2 3 4 5 6 7 8;1 2 3 4 5 6 7 8;1 2 3 4 5 6 7 8;1 2 3 4 5 6 7 8;1 2 3 4 5 6 7 8;1 2 3 4 5 6 7 8;1 2 3 4 5 6 7 8;]
%right_in=[4 5 6 7 8 9 1 2;4 5 6 7 8 9 1 2;4 5 6 7 8 9 1 2;4 5 6 7 8 9 1 2;4 5 6 7 8 9 1 2;4 5 6 7 8 9 1 2;4 5 6 7 8 9 1 2;4 5 6 7 8 9 1 2;]

left_res=zeros(size(left_in,1)+w,size(left_in,2)+w+dispmax);
right_res=zeros(size(right_in,1)+w,size(right_in,2)+w+dispmax);
 
left_res(wd2+1:end-wd2,wd2+1:end-wd2-dispmax)=left_in;
right_res(wd2+1:end-wd2,wd2+1:end-wd2-dispmax)=right_in;
left_res=uint8(left_res);
right_res=uint8(right_res);
figure
imshow(left_res)
figure
imshow(right_res)


for y=wd2+1:1:size(left_res,1)-wd2 %Left image row pix
    y %Y the scanline
    for x=wd2+1:1:size(left_res,2)-wd2-dispmax  %Left image col pix
        left=left_res(y-wd2:y+wd2,x-wd2:x+wd2); %Left image window w*w
        
        for disp=dispmin:1:dispmax
            right=right_res(y-wd2:y+wd2,x-wd2+disp:x+wd2+disp);
            SAD(disp)=sum(abs(left(:)-right(:))); %Take the sum of absolute difference
        end
            [SADval(y,x),depth(y-wd2,x-wd2)]=min(SAD(dispmin:dispmax));
    end
end

imagesc(depth)
% 
% 
% 
% xyzPoints = reconstructScene(depth,cameracalib.calibrationSession.CameraParameters);
% points3D = xyzPoints ./ 1000;
% ptCloud = pointCloud(points3D,'Color',leftimg_col);
% % Create a streaming point cloud viewer
% player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
% % Visualize the point cloud
% view(player3D, ptCloud);