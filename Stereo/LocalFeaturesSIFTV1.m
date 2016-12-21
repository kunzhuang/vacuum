clc;clear;close all;
%pack
ImageSource='Own'; %'Own' 'MB'
ImageSourceCrop=0;
Imgradient=0;
CannyEdge=1;
SobelEdge=0;
EdgeFeatures=0;
SIFT=1;
SIFTPointOnly=1;
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
%% SIFT match
if SIFT
%run('D:\Dropbox\Thesis\code\package\vlfeat-0.9.20-bin\vlfeat-0.9.20\toolbox\vl_setup')
[fa, da] = vl_sift(single(left_rectf_crop)) ; %plot(fa(1,:)',fa(2,:)','x')
[fb, db] = vl_sift(single(right_rectf_crop)) ;
[matches, scores] = vl_ubcmatch(da, db,1) ;

La=matches(1,:);Lb=fa(:,La); Lx=Lb(1,:);

LeftFeature=round([Lb(1,:)',Lb(2,:)']);


Ra=matches(2,:);Rb=fb(:,Ra);Rx=Rb(1,:);

RightFeature=round([Rb(1,:)',Rb(2,:)']);


left_in=left_rectf_crop;
right_in=right_rectf_crop;


RightFeatureMatch=RightFeature((LeftFeature(:,2)-RightFeature(:,2)==0&LeftFeature(:,2)-RightFeature(:,2)<=512),:);
LefttFeatureMatch=LeftFeature((LeftFeature(:,2)-RightFeature(:,2)==0&LeftFeature(:,2)-RightFeature(:,2)<=512),:);



LeftDisp=LefttFeatureMatch-RightFeatureMatch;
LeftDisp=[LefttFeatureMatch(:,1),LefttFeatureMatch(:,2),LeftDisp(:,1)];


SparseDisp=accumarray([LeftDisp(:,2),LeftDisp(:,1)],LeftDisp(:,3));
SparseDispT=zeros(size(left_in,1),size(left_in,2));
SparseDispT(1:size(SparseDisp,1),1:size(SparseDisp,2))=SparseDisp;
figure; showMatchedFeatures(left_in,right_in,LefttFeatureMatch,RightFeatureMatch);

figure
imshow(left_rectf_crop)
hold on
plot(LefttFeatureMatch(:,1)',LefttFeatureMatch(:,2)','x')


figure
imshow(right_rectf_crop)
hold on
plot(RightFeatureMatch(:,1)',RightFeatureMatch(:,2)','x')



ThreeDReconstruction(SparseDispT,leftimg_col,0,cameracalib.calibrationSession.CameraParameters)


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
%player3D = pcplayer([-3, 3], [0,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
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