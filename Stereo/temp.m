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
ImageSource='MB'; %'Own' 'MB'
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

 Left=load('DMapSUMMedL');
 Right=load('DMapSUMMedR');


%% 
Left.disparityMapSUMMed
Right.disparityMapSUMMed


LRRLcheck=1;

if LRRLcheck
    LefrCoord=repmat((1:Cols),Rows,1);RightCoord=repmat((1:Cols),Rows,1);
    CKL=RightCoord-Left.disparityMapSUMMed;
    %CKLLL=CKL;
    %CKL(CKL<=0)=1;
    CKLL=zeros(Rows,Cols);
    for Row=1:Rows
        for Col=1:Cols
            if CKL(Row,Col)<=0
                continue;
            else
                CKLL(Row,Col)=Right.disparityMapSUMMed(Row,CKL(Row,Col))+CKL(Row,Col);
            end
        end
    end
    CKLL=medfilt2(CKLL);
    CKLZ=zeros(Rows,Cols);
    for Col=1:Cols
        for Row=1:Rows
            if CKLL(Row,Col)==0
                continue;
            else
                CKLZ(Row,Col)=(abs(Col-CKLL(Row,Col))<500);
            end
        end
    end %Valid points after cross check
    %CKLLL(diff(CKLLL)==0)=0;
    
    %DisparitySpace=DisparitySpaceLeft.*CKLZ;
else
    %DisparitySpace=DisparitySpaceLeft;
end





xyzPoints = reconstructScene(DisparitySpaceLeft.disparityMapSUMMed.*CKLZ,CamCalib);

points3D = xyzPoints ./ 1000;
ptCloud = pointCloud(points3D,'Color',leftimg_col);
%Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
%Visualize the point cloud
show(player3D);
view(player3D, ptCloud)