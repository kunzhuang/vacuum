clc;clear;close all;tic;
%
ImageSource='Own'; %'Own' 'MB'
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
w=49; %even numbers only
disp_th=10;
Gaussianfilter=0;
Crosscheck=2;
Mode='Normal'; % 'Normal' 'SGM'
UniquenessThreshold=1;
DistanceThreshold=0;


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
        disparityRange = [0 64];
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
figure
imshow(left_rectf_crop)
figure
imshow(right_rectf_crop)

left_in=left_rectf_crop;
right_in=right_rectf_crop;

%%
switch Mode
    case 'SGM'
        left_in=[254*ones(size(left_in,1),disparityRange(1,2)) left_in ];
        right_in=[254*ones(size(right_in,1),disparityRange(1,2)) right_in ];
        disparityMap = disparity(left_in,right_in,'BlockSize',w,'DisparityRange',disparityRange,'UniquenessThreshold',UniquenessThreshold,'DistanceThreshold',DistanceThreshold);
        disparityMap=disparityMap(1:end,disparityRange(1,2)+1:end);
        disparityMap(disparityMap<=0)=0;
        figure
        imshow(255*disparityMap/max(max(disparityMap)),[1,255]);
        title('Disparity Map');
        colormap jet
        colorbar
        
        figure
        imagesc(255*disparityMap/max(max(disparityMap)))
        colormap jet
        colorbar

    case 'Normal'
        
        left_in=[254*ones(size(left_in,1),disparityRange(1,2)) left_in ];
        right_in=[254*ones(size(right_in,1),disparityRange(1,2)) right_in ];
        
        disparityMap = disparity(left_in,right_in,'BlockSize',w,'Method','BlockMatching','DisparityRange',disparityRange,'UniquenessThreshold',UniquenessThreshold,'DistanceThreshold',DistanceThreshold);    
        disparityMap=disparityMap(1:end,disparityRange(1,2)+1:end);
        figure
        imshow(255*disparityMap/max(max(disparityMap)),[1,255]);
        title('Disparity Map');
        colormap jet
        colorbar

        figure
        imagesc(255*disparityMap/max(max(disparityMap)))
        colormap jet
        colorbar
end

%%
if strcmp(ImageSource,'Own')
x1=172;y1=119; %2.5
x2=417;y2=345; %1.5
x3=425;y3=107; %3.6
xx=270;

d1=0.2*1333/disparityMap(y1,x1)
d2=0.2*1333/disparityMap(y2,x2)
d3=0.2*1333/disparityMap(y3,x3)

if ImageSourceCrop
    if Bilinearfilter
        left_res=Croped3DReconstruction(size(leftimg_col,1),size(leftimg_col,2),CropArea,bflt_img)
    else
        left_res=Croped3DReconstruction(size(leftimg_col,1),size(leftimg_col,2),CropArea,disparityMap);
    end
else left_res=disparityMap;
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


%% 
function left_res=Croped3DReconstruction(Rows,Cols,CropArea,depth)
%CropArea=[400 150 570 400];
left_res=zeros(Rows,Cols);
left_res(CropArea(2)+1:CropArea(2)+CropArea(4)+1,CropArea(1)+1:CropArea(1)+CropArea(3)+1)=depth;
end
