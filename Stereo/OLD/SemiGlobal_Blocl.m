clc;clear;close all;

cameracalib=load('D:\Dropbox\Thesis\code\Data\data\set1IR_LR\calibrationSessionLR.mat');

left_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters1;
right_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters2;

%leftimg=undistortImage(rgb2gray(imread('l.png')),left_cali);
%rightimg=undistortImage(rgb2gray(imread('r.png')),right_cali);

leftimg=rgb2gray(imread('l.png'));
rightimg=rgb2gray(imread('r.png'));

disparityRange = [64 1024];
window_size=9;

%Mode='Normal'; %Normal SGM
Mode='SGM';
Canny=0;
SIFT=1;

[left_rectf_crop, right_rectf_crop] = rectifyStereoImages(leftimg,rightimg,cameracalib.calibrationSession.CameraParameters);
[leftimg_col, rightimg_col] = rectifyStereoImages(imread('l.png'),imread('r.png'),cameracalib.calibrationSession.CameraParameters);


%%

% [Gmagl, left_rectf_crop] = imgradient(left_rectf_crop,'prewitt');
% [Gmagr, right_rectf_crop] = imgradient(right_rectf_crop,'prewitt');

% left_rectf_crop1=edge(left_rectf_crop,'Sobel',0);
% right_rectf_crop1=edge(right_rectf_crop,'Sobel',0);

%%
switch Canny
    case 1
        left_rectf_crop_Canny=edge(left_rectf_crop,'Canny',0.1)*254;
        right_rectf_crop_Canny=edge(right_rectf_crop,'Canny',0.1)*254;

        left_rectf_crop=left_rectf_crop_Canny ; %Replace current image with edge only image
        right_rectf_crop=right_rectf_crop_Canny;
end

%% Gauss filter
% left_rectf_crop=imgaussfilt(left_rectf_crop, 1);
% right_rectf_crop=imgaussfilt(right_rectf_crop, 1);


%% SIFT match
switch SIFT
    case 1
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

    left_rectf_crop_SIFT = logical(accumarray([int16(Lb(2,:))',int16(Lx)'],ones(size(Lx,2),1),(size(left_rectf_crop))));
    right_rectf_crop_SIFT= logical(accumarray([int16(Rb(2,:))',int16(Rx)'],ones(size(Lx,2),1),(size(left_rectf_crop))));
    left_rectf_crop=left_rectf_crop_SIFT ; %Replace current image with SIFT features
    right_rectf_crop=right_rectf_crop_SIFT;
end


%% 

% left_rectf_crop(left_rectf_Canny)=1;
% right_rectf_crop(right_rectf_Canny)=1;
% 
% left_rectf_crop(left_rectf_crop_SIFT)=1;
% right_rectf_crop(right_rectf_crop_SIFT)=1;

%%

%figure
%imshow(stereoAnaglyph(left_c,right_c));

%%
switch Mode
    case 'SGM'
        disparityMap = disparity(left_rectf_crop,right_rectf_crop,'BlockSize',window_size,'DisparityRange',disparityRange,'UniquenessThreshold',15,'DistanceThreshold',1);
        disparityMap(disparityMap<0)=0;
        figure
        imshow(disparityMap,disparityRange);
        title('Disparity Map');
        colormap jet
        colorbar
        
        figure
        imagesc(disparityMap)
        colormap jet
        colorbar

    case 'Normal'
        left_c=[254*ones(size(left_rectf_crop,1),disparityRange(1,2)) left_c ];
        right_c=[254*ones(size(left_rectf_crop,1),disparityRange(1,2)) right_c ];

        disparityMap = disparity(left_rectf_crop,right_rectf_crop,'BlockSize',window_size,'Method','BlockMatching','DisparityRange',disparityRange,'UniquenessThreshold',15,'DistanceThreshold',1);
        disparityMap=disparityMap(1:end,disparityRange(1,2)+1:end);
        disparityMap(disparityMap<0)=0;
        figure
        imshow(disparityMap,disparityRange);
        title('Disparity Map');
        colormap jet
        colorbar

        figure
        imagesc(disparityMap)
        colormap jet
        colorbar
end

xyzPoints = reconstructScene(disparityMap,cameracalib.calibrationSession.CameraParameters);
points3D = xyzPoints ./ 1000;
ptCloud = pointCloud(points3D,'Color',leftimg_col);
% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
% Visualize the point cloud
view(player3D, ptCloud);
