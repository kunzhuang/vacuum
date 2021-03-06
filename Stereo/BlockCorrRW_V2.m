clc;clear;close all;
%
ImageSource='MB'; %'Own' 'MB'
ImageSourceCrop=0;
Imgradient=0;
CannyEdge=0;
SobelEdge=0;
EdgeFeatures=0;
SIFT=0;
SIFTPointOnly=0;
Bilinearfilter=0;

%
% 0.26 to 4.1 meters
w=5; %even numbers only
disp_th=10;
Gaussianfilter=1;
Crosscheck=1;


%% Switch input type
switch ImageSource
    case 'Own'
        disparityRange = [64 1024];
        cameracalib=load('D:\Dropbox\Thesis\code\TestCode\V1\calibrationSessionLR.mat');
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
    left_rectf_crop_Canny=edge(left_rectf_crop,'Canny',0.1)*254;
    right_rectf_crop_Canny=edge(right_rectf_crop,'Canny',0.1)*254;
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
    left_rectf_crop=imgaussfilt(left_rectf_crop, 1);
    right_rectf_crop=imgaussfilt(right_rectf_crop, 1);
end
%% SIFT match
if SIFT
    run('D:\Dropbox\Thesis\code\package\vlfeat-0.9.20-bin\vlfeat-0.9.20\toolbox\vl_setup')
    [fa, da] = vl_sift(single(left_rectf_crop)) ; %plot(fa(1,:)',fa(2,:)','x')
    [fb, db] = vl_sift(single(right_rectf_crop)) ;
    [matches, scores] = vl_ubcmatch(da, db,1) ;
    
    La=matches(1,:);
    Lb=fa(:,La); Lx=Lb(1,:);
    % imshow(left_rectf_crop)
    % hold on
    % plot(Lb(1,:)',Lb(2,:)','x')
    
    % figure
    Ra=matches(2,:);
    Rb=fb(:,Ra);Rx=Rb(1,:);
    % imshow(right_rectf_crop)
    % hold on
    % plot(Rb(1,:)',Rb(2,:)','x')
    if SIFTPointOnly
        left_rectf_crop(left_rectf_crop_SIFT)=1;
        right_rectf_crop(right_rectf_crop_SIFT)=1;
    else
        left_rectf_crop = logical(accumarray([int16(Lb(2,:))',int16(Lx)'],ones(size(Lx,2),1),(size(left_rectf_crop))));
        right_rectf_crop= logical(accumarray([int16(Rb(2,:))',int16(Rx)'],ones(size(Lx,2),1),(size(left_rectf_crop))));
    end
end
%%
% figure
% imshow(left_rectf_crop)
% figure
% imshow(right_rectf_crop)


%Initialize variables
left_in=left_rectf_crop;
right_in=right_rectf_crop;

dispmax=disparityRange(1,2);dispmin=disparityRange(1,1);



Rows=size(right_in,1);
Cols=size(right_in,2);

Window=ones(w,w);
left_in_conv2=conv2(double(left_in),Window,'same');
right_in_conv2=conv2(double(right_in),Window,'same');
right_d_window=zeros(size(left_in,1),size(left_in,2),dispmax-dispmin+1);
left_d_window=zeros(size(left_in,1),size(left_in,2),dispmax-dispmin+1);
left_in_conv2_3DRepMap=repmat(left_in_conv2,1,1,dispmax-dispmin+1);
right_in_conv2_3DRepMap=repmat(right_in_conv2,1,1,dispmax-dispmin+1);
DisparitySpaceImageRightInLeftCoord=zeros(size(left_in,1),size(left_in,2),dispmax-dispmin+1);

for d_window=1:dispmax-dispmin+1
    d_window_val=dispmin:dispmax;
    if d_window_val(d_window)<size(right_in_conv2,2)
    right_d_window(:,:,d_window)=[zeros(size(right_in_conv2,1),d_window_val(d_window)),right_in_conv2(:,1:size(right_in_conv2,2)-d_window_val(d_window))];
    %Padding zeros to right image
    left_d_window(:,:,d_window)=[left_in_conv2(:,d_window_val(d_window)+1:end),zeros(size(right_in_conv2,1),d_window_val(d_window))];
    %Padding zeros to left image
    else
        right_d_window(:,:,d_window)=zeros(size(right_in_conv2,1),size(right_in_conv2,2));
        left_d_window(:,:,d_window)=zeros(size(right_in_conv2,1),size(right_in_conv2,2)); % set to zeros
    end
end

DisparitySpaceImageLeft=abs(left_in_conv2_3DRepMap(:,:,:)-right_d_window(:,:,:));
DisparitySpaceImageRight=abs(right_in_conv2_3DRepMap(:,:,:)-left_d_window(:,:,:));



[~, DisparitySpaceLeft]=min(DisparitySpaceImageLeft,[],3);
[~, DisparitySpaceRight]=min(DisparitySpaceImageRight,[],3);

%medfilt2
if Crosscheck
    LefrCoord=repmat((1:Cols),Rows,1);
    RightCoord=repmat((1:Cols),Rows,1);
    CKL=LefrCoord-DisparitySpaceLeft;
    CKL(CKL<=0)=1;
    for Row=1:Rows
        CKLL(Row,:)=DisparitySpaceRight(Row,CKL(Row,:))+CKL(Row,:);
    end
    
    
    CKLL=medfilt2(CKLL);
    
    CKLZ=zeros(Rows,Cols);
    
    for Col=1:Cols
        CKLZ(:,Col)=abs(Col-CKLL(:,Col))<disp_th;
    end %Valid points after cross check
    
    DisparitySpace=DisparitySpaceLeft.*CKLZ;
else
    DisparitySpace=DisparitySpaceLeft;
end

% figure
% imshow(255*DisparitySpaceLeft/max(max(DisparitySpaceLeft)),[1,255])
% colormap jet
% colorbar
% 
% figure
% imshow(255*DisparitySpaceRight/max(max(DisparitySpaceRight)),[1,255])
% colormap jet
% colorbar
% 
% figure
% imshow(255*DisparitySpace/max(max(DisparitySpace)),[1,255])
% colormap jet
% colorbar
figure
imagesc(255*DisparitySpaceLeft/max(max(DisparitySpaceLeft)))
colormap jet
colorbar

figure
imagesc(255*DisparitySpaceLeft/max(max(DisparitySpaceLeft)))
colormap jet
colorbar

figure
imagesc(255*DisparitySpace/max(max(DisparitySpace)))
colormap jet
colorbar

%% 3D reconstruct
CamCalib=cameracalib.calibrationSession.CameraParameters;
CropArea=[400 150 570 400];

left_r=zeros(size(leftimg_col,1),size(leftimg_col,2));
left_r(CropArea(2)+1:CropArea(2)+CropArea(4)+1,CropArea(1)+1:CropArea(1)+CropArea(3)+1)=DisparitySpace;

xyzPoints = reconstructScene(left_r,CamCalib);
points3D = xyzPoints ./ 1000;
ptCloud = pointCloud(points3D,'Color',leftimg_col);
%Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
%Visualize the point cloud
show(player3D);
view(player3D, ptCloud)

%% Bilinear filter
% Set bilateral filter parameters.
if Bilinearfilter
    w     = 5;       % bilateral filter half-width 5 3 0.1
    sigma = [2 0.1]; % bilateral filter standard deviations
    img = depth/max(max(depth));
    
    % Apply bilateral filter to each image.
    bflt_img = bfilter2(img,w,sigma)*max(max(depth));
    
    depth_normal=0.2*1333./depth;depth_normal(depth_normal==Inf|depth_normal>5)=0;
    depth_bi=0.2*1333./bflt_img;depth_bi(depth_bi==Inf|depth_bi>5)=0;
    
    figure
    surf(depth_normal)
    figure
    surf(depth_bi)
    
    xx=270;
    plot(depth_normal(xx,:))
    hold on
    plot(depth_bi(xx,:))
    
    figure
    imagesc(depth)
    figure
    imagesc(bflt_img)
    
    
    d4=0.2*1333/bflt_img(y1,x1)
    d5=0.2*1333/bflt_img(y2,x2)
    d6=0.2*1333/bflt_img(y3,x3)
end



%%
if strcmp(ImageSource,'Own')
    x1=172;y1=119; %2.5
    x2=417;y2=345; %1.5
    x3=425;y3=107; %3.6
    xx=270;
    depth=DisparitySpace;
    d1=0.2*1333/depth(y1,x1)
    d2=0.2*1333/depth(y2,x2)
    d3=0.2*1333/depth(y3,x3)
    
    if ImageSourceCrop
        if Bilinearfilter
            left_res=Croped3DReconstruction(size(leftimg_col,1),size(leftimg_col,2),CropArea,bflt_img)
        else
            left_res=Croped3DReconstruction(size(leftimg_col,1),size(leftimg_col,2),CropArea,depth);
        end
    else left_res=depth;
    end
    
    
    CamCalib=cameracalib.calibrationSession.CameraParameters;
    %reconstruction3D(left_res,CamCalib,leftimg_col);
    
    
    xyzPoints = reconstructScene(left_res,CamCalib);
    points3D = xyzPoints ./ 1000;
    ptCloud = pointCloud(points3D,'Color',leftimg_col);
    % Create a streaming point cloud viewer
    player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
    % Visualize the point cloud
    show(player3D);
    view(player3D, ptCloud)
    
    
end

%%
function left_res=Croped3DReconstruction(Rows,Cols,CropArea,depth)
%CropArea=[400 150 570 400];
left_res=zeros(Rows,Cols);
left_res(CropArea(2)+1:CropArea(2)+CropArea(4)+1,CropArea(1)+1:CropArea(1)+CropArea(3)+1)=depth;
end
%%
% function tmp=reconstruction3D(left_res,Caliberation,leftimg_col)
% xyzPoints = reconstructScene(left_res,Caliberation);
% points3D = xyzPoints ./ 1000;
% ptCloud = pointCloud(points3D,'Color',leftimg_col);
% % Create a streaming point cloud viewer
% player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
% % Visualize the point cloud
% show(player3D);
% view(player3D, ptCloud)




function disparityMap=ComputeDisparity(Rows, Cols, disparityRange,AggregatedCostArrary)
disparityMap=zeros(Rows,Cols);
for Col=1:Cols
    for Row=1:Rows
        [~,d]=min(AggregatedCostArrary(Row,Col,:));
        disparity = d+disparityRange(1,1)-1;
        disparityMap(Row,Col)=disparity;
    end
end
end



function disparityMap=ComputeDisparityNaN(Rows, Cols, disparityRange,AggregatedCostArrary)
disparityMap=zeros(Rows,Cols);
for Col=1:Cols
    for Row=1:Rows
        [~,d]=nanmin(AggregatedCostArrary(Row,Col,:));
        disparity = d+disparityRange(1,1)-1;
        disparityMap(Row,Col)=disparity;
    end
end
end
