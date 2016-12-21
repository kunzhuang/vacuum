clc;clear;close all;
%pack
ImageSource='Own'; %'Own' 'MB'
ImageSourceCrop=1;
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
left_rectf_crop=imgaussfilt(left_rectf_crop, 2);
right_rectf_crop=imgaussfilt(right_rectf_crop, 2);
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

%disparityRange=[2 4];
%Initialize variables
left_in=left_rectf_crop;
right_in=right_rectf_crop;

% PixelCost=load('PixelCost.mat');
% PixelCost=PixelCost.PixCost;
PathNum=16;
disparityRangeUP=disparityRange(2);
disparityRangeLOW=disparityRange(1);
disparityRangeLevels=disparityRange(2)-disparityRange(1)+1;



PixelCost=CalculatePixelCost(left_in,right_in,disparityRange);

Cols=size(left_in,2);Rows=size(left_in,1);
DispR=disparityRange(1):disparityRange(2);
AGCost=zeros(Rows,Cols,disparityRangeLevels);
AGCostSUM=zeros(Rows,Cols,disparityRangeLevels);


Path=PathDefine(PathNum,'ColFill');

SmallP=5;
LargeP=30;



% parfor p=1:6
% AGCost_Par(:,:,:,p)=uint8(PathScan(p,Path(p).row,Path(p).col,Path(p).cor,SmallP,LargeP,Cols,Rows,disparityRangeLevels,PixelCost));
% end
% for p=1:6
%     PP.(sprintf('Path_%i', p))=ComputeDisparity(Rows, Cols, disparityRange,AGCost_Par(:,:,:,p));
%     save('PathFile.mat','PP','-append')
%     AGCostSUM=AGCostSUM+AGCost_Par(:,:,:,p);
% end
% 
% parfor p=7:12
% AGCost_Par(:,:,:,p)=uint8(PathScan(p,Path(p).row,Path(p).col,Path(p).cor,SmallP,LargeP,Cols,Rows,disparityRangeLevels,PixelCost));
% end
% 
% for p=7:12
%     PP.(sprintf('Path_%i', p))=ComputeDisparity(Rows, Cols, disparityRange,AGCost_Par(:,:,:,p));
%     save('PathFile.mat','PP','-append')
%     AGCostSUM=AGCostSUM+AGCost_Par(:,:,:,p);
% end
% 
% parfor p=13:16
% AGCost_Par(:,:,:,p)=uint8(PathScan(p,Path(p).row,Path(p).col,Path(p).cor,SmallP,LargeP,Cols,Rows,disparityRangeLevels,PixelCost));
% end
% for p=13:16
%     PP.(sprintf('Path_%i', p))=ComputeDisparity(Rows, Cols, disparityRange,AGCost_Par(:,:,:,p));
%     save('PathFile.mat','PP','-append')
%     AGCostSUM=AGCostSUM+AGCost_Par(:,:,:,p);
% end


parfor p=1:16
AGCostSUM=AGCostSUM+PathScan(p,Path(p).row,Path(p).col,Path(p).cor,SmallP,LargeP,Cols,Rows,disparityRangeLevels,PixelCost);
end




%disparityMap=ComputeDisparity(size(left_in,1), size(left_in,2), disparityRange,AGCostSUM);

%AGCostHorizonSUM=zeros(size(left_in,1),size(left_in,2),disparityRange(2)-disparityRange(1)+1);


% for i=1:16
% figure (i)
% imshow(255*disparityMap(:,:,i)/max(max(disparityMap(:,:,i))),[1,255])
% colormap jet
% colorbar
% end

% temp=load('PathFile.mat');
% figure
% imshow(255*temp.PP.Path_1/max(max(temp.PP.Path_1)),[1,255])
% figure
% imshow(255*temp.PP.Path_2/max(max(temp.PP.Path_2)),[1,255])
% figure
% imshow(255*temp.PP.Path_3/max(max(temp.PP.Path_3)),[1,255])
% figure
% imshow(255*temp.PP.Path_4/max(max(temp.PP.Path_4)),[1,255])
% figure
% imshow(255*temp.PP.Path_5/max(max(temp.PP.Path_5)),[1,255])
% figure
% imshow(255*temp.PP.Path_6/max(max(temp.PP.Path_6)),[1,255])
% figure
% imshow(255*temp.PP.Path_7/max(max(temp.PP.Path_7)),[1,255])
% figure
% imshow(255*temp.PP.Path_8/max(max(temp.PP.Path_8)),[1,255])
% figure
% imshow(255*temp.PP.Path_9/max(max(temp.PP.Path_9)),[1,255])
% figure
% imshow(255*temp.PP.Path_10/max(max(temp.PP.Path_10)),[1,255])
% figure
% imshow(255*temp.PP.Path_11/max(max(temp.PP.Path_11)),[1,255])
% figure
% imshow(255*temp.PP.Path_12/max(max(temp.PP.Path_12)),[1,255])
% figure
% imshow(255*temp.PP.Path_13/max(max(temp.PP.Path_13)),[1,255])
% figure
% imshow(255*temp.PP.Path_14/max(max(temp.PP.Path_14)),[1,255])
% figure
% imshow(255*temp.PP.Path_15/max(max(temp.PP.Path_15)),[1,255])
% figure
% imshow(255*temp.PP.Path_16/max(max(temp.PP.Path_16)),[1,255])

% temp=AGCostSUM_Par(:,:,:,2);
% disparityMapSUM=ComputeDisparity(size(left_in,1), size(left_in,2), disparityRange,temp);
% figure
% imshow(255*disparityMapSUM/max(max(disparityMapSUM)),[1,255])



disparityMapSUM=ComputeDisparity(size(left_in,1), size(left_in,2), disparityRange,AGCostSUM);
figure
imshow(255*disparityMapSUM/max(max(disparityMapSUM)),[1,255])
PixelCostDisp=ComputeDisparity(size(left_in,1), size(left_in,2), disparityRange,PixelCost);
figure
imshow(255*PixelCostDisp/max(max(PixelCostDisp)),[1,255])

figure 
imshow(255*disparityMapSUM/max(max(disparityMapSUM)),[1,255])
colormap jet
colorbar


%imshow(255*AGCostHorizonSUM/max(max(disparityMapSUM)),[1,255])
figure
imagesc(255*disparityMapSUM/max(max(disparityMapSUM)))
colormap jet
colorbar

disparityMapSUMMed=medfilt2(disparityMapSUM);
figure
imagesc(255*disparityMapSUMMed/max(max(disparityMapSUMMed)))
colormap jet
colorbar



%% 3D reconstruct
CamCalib=cameracalib.calibrationSession.CameraParameters;
CropArea=[400 150 570 400];

left_r=zeros(size(leftimg_col,1),size(leftimg_col,2));
left_r(CropArea(2)+1:CropArea(2)+CropArea(4)+1,CropArea(1)+1:CropArea(1)+CropArea(3)+1)=disparityMapSUMMed;

xyzPoints = reconstructScene(left_r.disparityMapSUM,CamCalib);
points3D = xyzPoints ./ 1000;
ptCloud = pointCloud(points3D,'Color',leftimg_col);
%Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
%Visualize the point cloud
show(player3D);
view(player3D, ptCloud)
% 

%%


%%
function path=PathDefine(Path_num,Case) %Path_num=4 or 8 or 16
%Case 1
%Fill Col by Col downwards, left to right, left top corner first (cor==1),
%path 1 2 3 8 9 10 15 16
%%Fill Col by Col upwards, right to left, right buttom corner first (cor==2)
%path 4 5 6 7 11 12 13 14

%Case 2
%Fill Row by Row left to right, downwards, left top corner first (cor==1)
%path 1 2 3 4 9 10 11 12
%%Fill Row by Row  right to left, upwards, right buttom corner first (cor==2)
%path 5 6 7 8 13 14 15 16

if Case=='ColFill'
    if Path_num>=4
        path(1).i= 1; path(1).col=-1; path(1).row= 0; path(1).cor= 1;
        path(3).i= 3; path(3).col= 0; path(3).row=-1; path(3).cor= 1;
        path(5).i= 5; path(5).col= 1; path(5).row= 0; path(5).cor= 2;
        path(7).i= 7; path(7).col= 0; path(7).row= 1; path(7).cor= 2;
    end
    if Path_num>=8
        path(2).i= 2; path(2).col=-1; path(2).row=-1; path(2).cor= 1;
        path(4).i= 4; path(4).col= 1; path(4).row=-1; path(4).cor= 2;
        path(6).i= 6; path(6).col= 1; path(6).row= 1; path(6).cor= 2;
        path(8).i= 8; path(8).col=-1; path(8).row= 1; path(8).cor= 1;
    end
    if Path_num>=16
        path(9).i= 9; path(9).col= -2;path(9).row=-1; path(9).cor= 1;
        path(10).i=10;path(10).col=-1;path(10).row=-2;path(10).cor=1;
        path(11).i=11;path(11).col= 1;path(11).row=-2;path(11).cor=2;
        path(12).i=12;path(12).col= 2;path(12).row=-1;path(12).cor=2;
        path(13).i=13;path(13).col= 2;path(13).row= 1;path(13).cor=2;
        path(14).i=14;path(14).col= 1;path(14).row= 2;path(14).cor=2;
        path(15).i=15;path(15).col=-1;path(15).row= 2;path(15).cor=1;
        path(16).i=16;path(16).col=-2;path(16).row= 1;path(16).cor=1;
    end
else
    if Path_num>=4
        path(1).i= 1; path(1).col=-1; path(1).row= 0; path(1).cor= 1;
        path(3).i= 3; path(3).col= 0; path(3).row=-1; path(3).cor= 1;
        path(5).i= 5; path(5).col= 1; path(5).row= 0; path(5).cor= 2;
        path(7).i= 7; path(7).col= 0; path(7).row= 1; path(7).cor= 2;
    end
    if Path_num>=8
        path(2).i= 2; path(2).col=-1; path(2).row=-1; path(2).cor= 1;
        path(4).i= 4; path(4).col= 1; path(4).row=-1; path(4).cor= 1;
        path(6).i= 6; path(6).col= 1; path(6).row= 1; path(6).cor= 2;
        path(8).i= 8; path(8).col=-1; path(8).row= 1; path(8).cor= 2;
    end
    if Path_num>=16
        path(9).i= 9; path(9).col= -2;path(9).row=-1; path(9).cor= 1;
        path(10).i=10;path(10).col=-1;path(10).row=-2;path(10).cor=1;
        path(11).i=11;path(11).col= 1;path(11).row=-2;path(11).cor=1;
        path(12).i=12;path(12).col= 2;path(12).row=-1;path(12).cor=1;
        path(13).i=13;path(13).col= 2;path(13).row= 1;path(13).cor=2;
        path(14).i=14;path(14).col= 1;path(14).row= 2;path(14).cor=2;
        path(15).i=15;path(15).col=-1;path(15).row= 2;path(15).cor=2;
        path(16).i=16;path(16).col=-2;path(16).row= 1;path(16).cor=2;
    end
end

end


%%  Pixelwise Cost  Using A pixel Dimissilarity measure that is inseneitive to image sampling, Stan Birchfield and Carlo Tomasi
%Note xl=xl, xr=xl-dispairt, if xl is fixed -> disparity is a variable -> xl is variable
%                            if xl is moving along a row, xl is a variable -> disparity is a variable -> xl is variable
function PixCost=CalculatePixelCost(LeftImg,RightImg,disparityRange)

DisparityArrary=disparityRange(1):disparityRange(2);
LeftImgZPad=[zeros(size(LeftImg,1),disparityRange(2)+1),LeftImg,zeros(size(LeftImg,1),1)]; % Zero pad left image + disparityRange(2)(1024) +1 right +1
RightImgZPad=[zeros(size(LeftImg,1),disparityRange(2)+1),RightImg,zeros(size(LeftImg,1),1)];% Zero pad right image + disparityRange(2)(1024) +1 right +1

z=zeros(size(LeftImg,1),1,size(DisparityArrary,2)); % Creating a zero matrix
LeftValueOnCol=zeros(size(LeftImg,1),1,size(DisparityArrary,2));RightValueOnCol=zeros(size(LeftImg,1),1,size(DisparityArrary,2));
RightValueOnColT3_Raw=zeros(size(LeftImg,1),3,size(DisparityArrary,2));RightValueOnColT3_Cal=zeros(size(LeftImg,1),3,size(DisparityArrary,2));
LeftValueOnColT3_Raw=zeros(size(LeftImg,1),3,size(DisparityArrary,2));LeftValueOnColT3_Cal=zeros(size(LeftImg,1),3,size(DisparityArrary,2)); %Allocate everything
PixCost=zeros(size(LeftImg,1),size(LeftImg,2),size(DisparityArrary,2));

for Col=1:size(LeftImg,2)
    PercdntDone(Col,size(LeftImg,2),'PixelCost')
    LeftValueOnCol(:,1,:)=repmat(LeftImgZPad(:,disparityRange(2)+1+Col),[1,1,size(DisparityArrary,2)]); %Create a 3D matrix 1x1xsize(DisparityArrary,2), 1x1 from input image value at Col,Constant on Col
    %RightValueOnCol(:,1,:)=repmat(RightImg(:,Col),[1,1,size(DisparityArrary,2)]);
    LeftValueOnColT3_Raw(:,:,:)=repmat(LeftImgZPad(:,disparityRange(2)+1+Col-1:disparityRange(2)+1+Col+1),[1,1,size(DisparityArrary,2)]); %Constant on Col, xl=xl, xr=xl-disparity
    
    for dd=1:size(DisparityArrary,2)
        RightValueOnColT3_Raw(:,:,dd)=RightImgZPad(:,(disparityRange(2)+1+Col-DisparityArrary(dd))-1:(disparityRange(2)+1+Col-DisparityArrary(dd))+1); %(disparityRange(2)+1+Col=current Col location after zero padding
        % create a Row x 3 x disparityRange(2)+1 matrix to store image values on left/right dispairt-1, disparity, disparity+1
        RightValueOnCol(:,:,dd)=RightImgZPad(:,(disparityRange(2)+1+Col-DisparityArrary(dd))); % Get right image value xl-disparity,xl=xl, xr=xl-disparity
    end
    
    RightValueOnColT3_Cal(:,:,:)=[RightValueOnColT3_Raw(:,1,:)*0.5+RightValueOnColT3_Raw(:,2,:)*0.5,...
                                  RightValueOnColT3_Raw(:,2,:),...
                                  RightValueOnColT3_Raw(:,2,:)*0.5+RightValueOnColT3_Raw(:,3,:)*0.5]; %0.5*(Ir(xr)+Ir(xr-1)), (Ir(xr)), 0.5*(Ir(xr)+Ir(xr+1))
    
    LeftValueOnColT3_Cal(:,:,:)=[LeftValueOnColT3_Raw(:,1,:)*0.5+LeftValueOnColT3_Raw(:,2,:)*0.5,...
                                 LeftValueOnColT3_Raw(:,2,:),...
                                 LeftValueOnColT3_Raw(:,2,:)*0.5+LeftValueOnColT3_Raw(:,3,:)*0.5];%0.5*(Il(xl)+Il(xl-1)), (Ir(xl)), 0.5*(Ir(xl)+Il(xl+1))
    
    RightMax=max(RightValueOnColT3_Cal(:,:,:),[],2); %ImaxR
    RightMin=min(RightValueOnColT3_Cal(:,:,:),[],2);%IminR
    Left_bar=max([z,LeftValueOnCol(:,:,:)-RightMax,RightMin-LeftValueOnCol(:,:,:)],[],2);%d(xl,xr,Il,Ir)
    
    LeftMax=max(LeftValueOnColT3_Cal(:,:,:),[],2); %ImaxL
    LeftMin=min(LeftValueOnColT3_Cal(:,:,:),[],2); %IminL
    Right_bar=max([z,RightValueOnCol(:,:,:)-LeftMax,LeftMin-RightValueOnCol(:,:,:)],[],2); %d(xl,xr,Il,Ir)
    
    Pix_Cost=min([Left_bar,Right_bar],[],2); %min (d(xl,xr,Il,Ir) d(xl,xr,Il,Ir))
    
    PixCost(:,Col,:)=Pix_Cost;
end
end

%% ComputeDisparity find the min energy along (Row,Col,Disparity) ->min value on Disparity axis
% function disparityMap=ComputeDisparity2(Rows, Cols, disparityRange,AggregatedCostArrary)
% for p=1:16
% for Col=1:Cols
%     for Row=1:Rows
%         [~,d]=min(AggregatedCostArrary(Row,Col,:,p));
%         disparity = d+disparityRange(1,1)-1;
%         disparityMap(Row,Col,p)=disparity;
%     end
% end
% end
% end


function disparityMap=ComputeDisparity(Rows, Cols, disparityRange,AggregatedCostArrary)
disparityMap=zeros(Rows, Cols);
for Col=1:Cols
    for Row=1:Rows
        [~,d]=min(AggregatedCostArrary(Row,Col,:));
        disparity = d+disparityRange(1,1)-1;
        disparityMap(Row,Col)=disparity;
    end
end
end

function AGCost=PathScan(p,RR,CC,Corn,SmallP,LargeP,Cols,Rows,disparityRangeLevels,PixelCost)
AGCost=zeros(Rows,Cols,disparityRangeLevels);

RemoveNeighbourhoodDisp=zeros(disparityRangeLevels,disparityRangeLevels);
for C_construct=1:disparityRangeLevels
    RemoveNeighbourhoodDisp(C_construct:C_construct+2,C_construct)=1;
end
RemoveNeighbourhoodDisp=logical(RemoveNeighbourhoodDisp);


%Create a matrix like
%      1     0     0     0     0
%      1     1     0     0     0
%      1     1     1     0     0
%      0     1     1     1     0
%      0     0     1     1     1
%      0     0     0     1     1
%      0     0     0     0     1

if Corn==1
    for Col=1:Cols
        PercdntDone(Col,Cols,sprintf('Path_%i', p),20)
        for Row=1:Rows
            LrLastRow=Row + RR ;LrLastCol=Col + CC;
            if (LrLastRow <= 0 | LrLastRow >= Rows | LrLastCol<= 0| LrLastCol>= Cols) %check borders
                AGCost(Row,Col,:)  =PixelCost(Row,Col,:);
            else
                LrLastMin=min(AGCost(LrLastRow,LrLastCol,:)); %Calculate global min on LrLast->Lr(p-r,k)
                AGCost(Row,Col,:)=PixelCost(Row,Col,:);
                
                LrLast=reshape(AGCost(LrLastRow,LrLastCol,:),[],1);
                
                LrLastDMone=[AGCost(LrLastRow,LrLastCol,1);reshape(AGCost(LrLastRow,LrLastCol,1:disparityRangeLevels-1),[],1)]+SmallP;
                
                LrLastDPone=[reshape(AGCost(LrLastRow,LrLastCol,2:disparityRangeLevels),[],1);AGCost(LrLastRow,LrLastCol,disparityRangeLevels)]+SmallP;
                
                LrLastMinOthers=[256*ones(1,disparityRangeLevels);repmat(reshape(AGCost(LrLastRow,LrLastCol,:),[],1),1,disparityRangeLevels);256*ones(1,disparityRangeLevels)];
                LrLastMinOthers(RemoveNeighbourhoodDisp)=256;
                LrLastMinOthers = min(LrLastMinOthers);
                LrLastMinOthers=(LrLastMinOthers+LargeP)';
                
                
                LrLast=min([LrLast,LrLastDMone,LrLastDPone,LrLastMinOthers],[],2)-LrLastMin;
                LrLast=reshape(LrLast,1,1,disparityRangeLevels);
                AGCost(Row,Col,:)= AGCost(Row,Col,:)+ LrLast;
            end
        end
    end
    %PP.(sprintf('Path_%i', p))=ComputeDisparity(Rows, Cols, disparityRange,AGCost);
    %save('PathFile.mat','PP','-append')
    %AGCostSUM=AGCost+AGCostSUM;
end


if Corn==2
    for Col=Cols:-1:1
        PercdntDone(Col,Cols,sprintf('Path_%i', p),20)
        for Row=Rows:-1:1
            LrLastRow=Row + RR ;LrLastCol=Col + CC;
            if (LrLastRow <= 0 | LrLastRow >= Rows | LrLastCol<= 0| LrLastCol>= Cols) %check borders
                AGCost(Row,Col,:)  =PixelCost(Row,Col,:);
            else
                LrLastMin=min(AGCost(LrLastRow,LrLastCol,:)); %Calculate global min on LrLast->Lr(p-r,k)
                AGCost(Row,Col,:)=PixelCost(Row,Col,:);
                
                LrLast=reshape(AGCost(LrLastRow,LrLastCol,:),[],1);
                
                LrLastDMone=[AGCost(LrLastRow,LrLastCol,1);reshape(AGCost(LrLastRow,LrLastCol,1:disparityRangeLevels-1),[],1)]+SmallP;
                
                LrLastDPone=[reshape(AGCost(LrLastRow,LrLastCol,2:disparityRangeLevels),[],1);AGCost(LrLastRow,LrLastCol,disparityRangeLevels)]+SmallP;
                
                LrLastMinOthers=[256*ones(1,disparityRangeLevels);repmat(reshape(AGCost(LrLastRow,LrLastCol,:),[],1),1,disparityRangeLevels);256*ones(1,disparityRangeLevels)];
                LrLastMinOthers(RemoveNeighbourhoodDisp)=256;
                LrLastMinOthers = min(LrLastMinOthers);
                LrLastMinOthers=(LrLastMinOthers+LargeP)';
                
                
                LrLast=min([LrLast,LrLastDMone,LrLastDPone,LrLastMinOthers],[],2)-LrLastMin;
                LrLast=reshape(LrLast,1,1,disparityRangeLevels);
                AGCost(Row,Col,:)= AGCost(Row,Col,:)+ LrLast;
            end
        end
    end
end
end
