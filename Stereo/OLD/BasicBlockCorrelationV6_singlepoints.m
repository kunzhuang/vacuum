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
% leftimg_col=imcrop(leftimg_col,[300 150 670 300]);
% rightimg_col=imcrop(rightimg_col,[300 150 670 300]);



% [Gmagl, left_rectf_crop] = imgradient(left_rectf_crop,'prewitt');
% [Gmagr, right_rectf_crop] = imgradient(right_rectf_crop,'prewitt');

% left_rectf_crop1=edge(left_rectf_crop,'Sobel',0.5);
% right_rectf_crop1=edge(right_rectf_crop,'Sobel',0.5);
% 
% left_rectf_crop1=edge(left_rectf_crop,'Canny',0.2);
% right_rectf_crop1=edge(right_rectf_crop,'Canny',0.2);
% 
% 
% left_rectf_crop=imgaussfilt(left_rectf_crop, 1);
% right_rectf_crop=imgaussfilt(right_rectf_crop, 1);


% 
% left_rectf_crop(left_rectf_crop1)=1;
% right_rectf_crop(right_rectf_crop1)=1;

disparityRange = [64 1024];
% 0.26 to 4.1 meters

% figure
% imshow(left_rectf_crop)
% figure
% imshow(right_rectf_crop)


left_in=left_rectf_crop;
right_in=right_rectf_crop;

%Initialize variables
w=49; %even numbers only
dispmax=disparityRange(1,2);dispmin=disparityRange(1,1);
% mod(w,2)
w=w-1; %left+center+right
wd2=w/2;

%Resize images
left_res=zeros(size(left_in,1)+w,size(left_in,2)+w+dispmax);
right_res=zeros(size(right_in,1)+w,size(right_in,2)+w+dispmax);
 
left_res(wd2+1:end-wd2,wd2+1+dispmax:end-wd2)=left_in;
right_res(wd2+1:end-wd2,wd2+1+dispmax:end-wd2)=right_in;
left_res=uint8(left_res);right_res=uint8(right_res);

% figure
% imshow(left_res)
% figure
% imshow(right_res)

SAD=zeros(1,dispmax-dispmin);


%x=1621;y=290; % 2.6m  102
%x=1847;y=501; % 1.5m  177
%x=1888;y=296; % 3.6m  73
%x=2235;y=417; % 2.5m  106
x=2469;y=613; %1.1m  238


figure 
imshow(left_res(y-wd2:y+wd2,x-wd2:x+wd2))

figure 
left=left_res(y-wd2:y+wd2,x-wd2:x+wd2); %Left image window w*w
       for disp=dispmin:1:dispmax
           if x-wd2-disp>wd2 %ignore black region on the left, right bound is constrainted by x
              right=right_res(y-wd2:y+wd2,x-wd2-disp:x+wd2-disp);              
              SAD(disp)=sum(sum(abs(left - right))); %Take the sum of absolute difference
              plot(SAD)
              %SSD(disp)=SAD.^2;
           end
       end
            [temp_max]=max(SAD(dispmin:dispmax));
            
            [temp,depth]=min(SAD(dispmin:dispmax));
            
            
depth=depth+dispmin-1

figure
imshow(right_res(y-wd2:y+wd2,x-wd2-depth:x+wd2-depth))
figure
plot(SAD)
xlabel('x-axis coord');ylabel('SAD score');
distance=0.2*1333/depth
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