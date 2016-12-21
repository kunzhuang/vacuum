% if LRRLcheck
%     LefrCoord=repmat((1:Cols),Rows,1);RightCoord=repmat((1:Cols),Rows,1);
%     CKL=RightCoord-DisparitySpaceLeft.disparityMapSUMMed;
%     %CKLLL=CKL;
%     %CKL(CKL<=0)=1;
%     CKLL=zeros(Rows,Cols);
%     for Row=1:Rows
%         CKLL(Row,:)=DisparitySpaceLeft.disparityMapSUMMed(Row,CKL(Row,:))+CKL(Row,:);
%     end
%     CKLL=medfilt2(CKLL);
%     CKLZ=zeros(Rows,Cols);
%     for Col=1:Cols
%         CKLZ(:,Col)=(abs(Col-CKLL(:,Col))<disp_th);
%     end %Valid points after cross check
%     %CKLLL(diff(CKLLL)==0)=0;
%     
%     %DisparitySpace=DisparitySpaceLeft.*CKLZ;
% else
%     %DisparitySpace=DisparitySpaceLeft;
% end

%pack
ImageSource='Own'; %'Own' 'MB'
ImageSourceCrop=0;
Imgradient=0;
CannyEdge=0;
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

 Left=load('DMapSUML');
 Right=load('DMapSUMR');


LRRL=LRRLConsistency(Left.disparityMapSUM,Right.disparityMapSUM,3);


ThreeDReconstruction(Left.disparityMapSUM,leftimg_col,0,cameracalib.calibrationSession.CameraParameters,LRRL)

%%
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
points3D (:,:,1)=points3D (:,:,1)*-1;
points3D (:,:,2)=points3D (:,:,2)*-1;
ptCloud = pointCloud(points3D,'Color',ColImg);
%Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [0,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
%Visualize the point cloud
pcshow(ptCloud,'VerticalAxis', 'y', 'VerticalAxisDir', 'Up')
xlabel('X(m)')
ylabel('Y(m)')
zlabel('Z depth(m)')
title('Point Cloud 3D reconstruction from stereo matching')
axis([-1.5 1.5 -0.25 1 0 5])
%view(player3D, ptCloud)
% 
end
%% Left to right, right to left consistency check
function CKLR= LRRLConsistency(LeftDisp,RightDisp,dis_th)
%Input left and right disparity map, and displary threshold
%Return realiable points
%Good points 1, unreliable points 0
Cols=size(LeftDisp,2);Rows=size(LeftDisp,1);
LeftCoord=repmat((1:Cols),Rows,1);
CKL=LeftCoord-LeftDisp; %Find right coordinate(col) at same row
CKR=zeros(Rows,Cols);
for Row=1:Rows
    for Col=1:Cols
        if CKL(Row,Col)<=0
            continue;
        else
            CKR(Row,Col)=RightDisp(Row,CKL(Row,Col))+CKL(Row,Col); %Find the corresponding right corrdinate(col)+right image disparity, assign this value to CKR(Row,Col)
        end
    end
end
CKR=medfilt2(CKR);
CKLR=zeros(Rows,Cols);
for Col=1:Cols
    for Row=1:Rows
        if CKR(Row,Col)==0
            continue;
        else
            CKLR(Row,Col)=(abs(Col-CKR(Row,Col))<=dis_th);
        end
    end
end
end