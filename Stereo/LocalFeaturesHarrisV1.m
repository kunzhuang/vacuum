clc;clear;close all;
%pack
ImageSource='MB'; %'Own' 'MB'
ImageSourceCrop=0;
Imgradient=0;
CannyEdge=1;
SobelEdge=0;
EdgeFeatures=0;
SIFT=0;
SIFTPointOnly=0;
Bilinearfilter=0;
Gaussianfilter=0;



%% Switch input type
switch ImageSource
    case 'Own'
        disparityRange = [64 512];%0.26 to 4.1 meters
        cameracalib=load('calibrationSessionLR.mat');
        left_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters1;
        right_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters2;
        leftimg=rgb2gray(imread('l.png'));rightimg=rgb2gray(imread('r.png'));
        %leftimg=undistortImage(rgb2gray(imread('l.png')),left_cali);
        %rightimg=undistortImage(rgb2gray(imread('r.png')),right_cali);
        [left_rectf_crop, right_rectf_crop] = rectifyStereoImages(leftimg,rightimg,cameracalib.calibrationSession.CameraParameters);
        [leftimg_col, rightimg_col] = rectifyStereoImages(imread('l.png'),imread('r.png'),cameracalib.calibrationSession.CameraParameters);
    case 'MB'
        disparityRange = [1 64];
        %leftimg=rgb2gray(imread('im0.png'));rightimg=rgb2gray(imread('im1.png'));
        left_rectf_crop=rgb2gray(imread('ted0.png'));right_rectf_crop=rgb2gray(imread('ted1.png'));
end

if ImageSourceCrop
       CropArea=[400 150 570 400];
       left_rectf_crop=imcrop(left_rectf_crop,CropArea);
       right_rectf_crop=imcrop(right_rectf_crop,CropArea);
%        leftimg_col=imcrop(leftimg_col,[300 150 670 300]);
%        rightimg_col=imcrop(rightimg_col,[300 150 670 300]);
end


%% Gradient magnitude and direction of an image
if Imgradient
   [Gmagl, left_rectf_crop] = imgradient(left_rectf_crop,'prewitt');
   [Gmagr, right_rectf_crop] = imgradient(right_rectf_crop,'prewitt');
end

%% SobelEdge
if SobelEdge
   left_rectf_crop_Sobel=edge(left_rectf_crop,'Sobel',0);
   right_rectf_crop_Sobel=edge(right_rectf_crop,'Sobel',0);
end
%% CannyEdge
if CannyEdge
   left_rectf_crop_Canny=edge(left_rectf_crop,'Canny',0.1);
   right_rectf_crop_Canny=edge(right_rectf_crop,'Canny',0.1);
end
%% EdgeFeatures only?
if EdgeFeatures
    if CannyEdge
       left_rectf_crop=left_rectf_crop_Canny;
       right_rectf_crop=right_rectf_crop_Canny;
    else
       left_rectf_crop=left_rectf_crop_Sobel;
       right_rectf_crop=left_rectf_crop_Sobel;
    end

end
%% Gauss filter
if Gaussianfilter
left_rectf_crop=imgaussfilt(left_rectf_crop, 2);
right_rectf_crop=imgaussfilt(right_rectf_crop, 2);
end
%% 

% left_rectf_crop(left_rectf_crop_Canny)=1;
% right_rectf_crop(right_rectf_crop_Canny)=1;
%Initialize variables
left_in=left_rectf_crop;
right_in=right_rectf_crop;

%% Harris feature
Harris=1;
if Harris
% figure
% imshow(left_in)
% figure
% imshow(right_in)

Cols=size(left_in,2);
Rows=size(left_in,1);
points1 =detectHarrisFeatures(left_in);
% points2 =detectHarrisFeatures(right_in);
% [features1,valid_points1] = extractFeatures(left_in,points1);
% [features2,valid_points2] = extractFeatures(right_in,points2);
% indexPairs = matchFeatures(features1,features2);
% matchedPoints1 = valid_points1(indexPairs(:,1),:);
% matchedPoints2 = valid_points2(indexPairs(:,2),:);
% figure; showMatchedFeatures(left_in,right_in,matchedPoints1,matchedPoints2);
% matchedPointsLoc1=uint16(matchedPoints1.Location);
% matchedPointsLoc2=uint16(matchedPoints2.Location);
% LeftDisp=[matchedPointsLoc1(:,1)-matchedPointsLoc2(:,1)];
% %matchedPointsLoc1=[matchedPointsLoc1 LeftDisp(:,1)];
% SparseDisp=accumarray([matchedPointsLoc1(:,2),matchedPointsLoc1(:,1)],LeftDisp);
% SparseDispT=zeros(size(left_in,1),size(left_in,2));
% SparseDispT(1:size(SparseDisp,1),1:size(SparseDisp,2))=SparseDisp;


LeftCorners=round(points1.Location);
RightCorners=zeros(size(LeftCorners,1),2);
THRESH = 0.70;       % correlation score threshold
WinSize=5;


for i=1:size(LeftCorners,1)
    Lx = LeftCorners(i,1);Ly = LeftCorners(i,2);
    M = floor(WinSize/2);
    if Lx<=M || Lx>Cols-M     
        continue;  
    end
    if Ly<=M || Ly>Rows-M     
        continue;  
    end
        T = left_in(Ly-M:Ly+M, Lx-M:Lx+M); % Generate a template
        C = normxcorr2(T,right_in);   % NCC Normalized cross correlation
    cmax = max(C(:));
    if cmax < THRESH     
        continue;  
    end
    [Ry,Rx] = find(C==cmax);
    Ry = Ry-M;Rx = Rx-M;
 
    RightCorners(i,:)=[Rx,Ry];
end

RightCorners=double(RightCorners);

RightCornerMatch=RightCorners((RightCorners(:,2)-LeftCorners(:,2)==0&RightCorners(:,2)-LeftCorners(:,2)<=512),:);
LeftCornerMatch=LeftCorners((RightCorners(:,2)-LeftCorners(:,2)==0&RightCorners(:,2)-LeftCorners(:,2)<=512),:);
LeftDisp=LeftCornerMatch-RightCornerMatch;
LeftDisp=[LeftCornerMatch(:,1),LeftCornerMatch(:,2),LeftDisp(:,1)];


SparseDisp=accumarray([LeftDisp(:,2),LeftDisp(:,1)],LeftDisp(:,3));
SparseDispT=zeros(size(left_in,1),size(left_in,2));
SparseDispT(1:size(SparseDisp,1),1:size(SparseDisp,2))=SparseDisp;
figure; showMatchedFeatures(left_in,right_in,LeftCornerMatch,RightCornerMatch);


%ThreeDReconstruction(SparseDispT,leftimg_col,0,cameracalib.calibrationSession.CameraParameters)
end

%% 3D reconstruct
function ThreeDReconstruction(DMap,ColImg,Crop,CamCalib,unreliable)
if Crop 
%CamCalib=cameracalib.calibrationSession.CameraParameters;
CropArea=[400 150 570 400];
DDmap=zeros(size(DMap,1),size(DMap,2));
DDmap(CropArea(2)+1:CropArea(2)+CropArea(4)+1,CropArea(1)+1:CropArea(1)+CropArea(3)+1)=DMap;
else
    DDmap=DMap;
end
% 
% left_r=zeros(size(leftimg_col,1),size(leftimg_col,2));
% left_r(CropArea(2)+1:CropArea(2)+CropArea(4)+1,CropArea(1)+1:CropArea(1)+CropArea(3)+1)=disparityMapSUMLL;
% xyzPoints = reconstructScene(left_r,CamCalib);

if exist('unreliable','var')
    DDmap=DDmap.*unreliable;
end

xyzPoints = reconstructScene(DDmap,CamCalib);
points3D = xyzPoints ./ 1000;
ptCloud = pointCloud(points3D,'Color',ColImg);
%Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [0,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
%Visualize the point cloud
pcshow(ptCloud,'VerticalAxis', 'y', 'VerticalAxisDir', 'Down')
xlabel('X(m)')
ylabel('Y(m)')
zlabel('Z depth(m)')
title('Point Cloud 3D reconstruction from stereo matching')
axis([-1.5 1.5 -1 0.25 0 5])
%view(player3D, ptCloud)
% 
end


%% 