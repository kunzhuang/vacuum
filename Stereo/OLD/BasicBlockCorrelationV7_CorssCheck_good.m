clc;clear;close all;
%run('D:\Dropbox\Thesis\code\package\vlfeat-0.9.20-bin\vlfeat-0.9.20\toolbox\vl_setup')
cameracalib=load('D:\Dropbox\Thesis\code\TestCode\V1\calibrationSessionLR.mat');

left_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters1;
right_cali=cameracalib.calibrationSession.CameraParameters.CameraParameters2;

%leftimg=undistortImage(rgb2gray(imread('l.png')),left_cali);
%rightimg=undistortImage(rgb2gray(imread('r.png')),right_cali);
leftimg=rgb2gray(imread('im0.png'));
rightimg=rgb2gray(imread('im1.png'));

disparityRange = [3 256];
%disparityRange = [64 1024];
% 0.26 to 4.1 meters
w=5; %even numbers only
disp_th=10;
[left_rectf_crop, right_rectf_crop] = rectifyStereoImages(leftimg,rightimg,cameracalib.calibrationSession.CameraParameters);
[leftimg_col, rightimg_col] = rectifyStereoImages(imread('l.png'),imread('r.png'),cameracalib.calibrationSession.CameraParameters);



% left_rectf_crop=imcrop(left_rectf_crop,[300 150 670 300]);
% right_rectf_crop=imcrop(right_rectf_crop,[300 150 670 300]);
% leftimg_col=imcrop(leftimg_col,[300 150 670 300]);
% rightimg_col=imcrop(rightimg_col,[300 150 670 300]);

%%

% [Gmagl, left_rectf_crop] = imgradient(left_rectf_crop,'prewitt');
% [Gmagr, right_rectf_crop] = imgradient(right_rectf_crop,'prewitt');

% left_rectf_crop1=edge(left_rectf_crop,'Sobel',0);
% right_rectf_crop1=edge(right_rectf_crop,'Sobel',0);

%%
% left_rectf_crop_Canny=edge(left_rectf_crop,'Canny',0.1)*254;
% right_rectf_crop_Canny=edge(right_rectf_crop,'Canny',0.1)*254;



%% Gauss filter
% left_rectf_crop=imgaussfilt(left_rectf_crop, 1);
% right_rectf_crop=imgaussfilt(right_rectf_crop, 1);




%% SIFT match
% [fa, da] = vl_sift(single(left_rectf_crop)) ; %plot(fa(1,:)',fa(2,:)','x')
% [fb, db] = vl_sift(single(right_rectf_crop)) ;
% [matches, scores] = vl_ubcmatch(da, db,1) ;
% 
% La=matches(1,:);
% Lb=fa(:,La); Lx=Lb(1,:);
% % imshow(left_rectf_crop)
% % hold on
% % plot(Lb(1,:)',Lb(2,:)','x')
% 
% % figure
% Ra=matches(2,:);
% Rb=fb(:,Ra);Rx=Rb(1,:);
% % imshow(right_rectf_crop)
% % hold on
% % plot(Rb(1,:)',Rb(2,:)','x')
% 
% left_rectf_crop_SIFT = logical(accumarray([int16(Lb(2,:))',int16(Lx)'],ones(size(Lx,2),1),(size(left_rectf_crop))));
% right_rectf_crop_SIFT= logical(accumarray([int16(Rb(2,:))',int16(Rx)'],ones(size(Lx,2),1),(size(left_rectf_crop))));

%% 
% left_rectf_crop(left_rectf_Canny)=1;
% right_rectf_crop(right_rectf_Canny)=1;
% 
% left_rectf_crop(left_rectf_crop_SIFT)=1;
% right_rectf_crop(right_rectf_crop_SIFT)=1;

%% 


figure
imshow(left_rectf_crop)
figure
imshow(right_rectf_crop)


left_in=leftimg;%left_rectf_crop;
right_in=rightimg;%right_rectf_crop;

%Initialize variables

dispmax=disparityRange(1,2);dispmin=disparityRange(1,1);
% mod(w,2)
w=w-1; %left+center+right
wd2=w/2;

%Resize images

left_res=zeros(size(left_in,1)+w,size(left_in,2)+w);
right_res=zeros(size(right_in,1)+w,size(right_in,2)+w);



left_res(wd2+1:end-wd2,wd2+1:end-wd2)=left_in;
right_res(wd2+1:end-wd2,wd2+1:end-wd2)=right_in;
left_res=uint8(left_res);right_res=uint8(right_res);


SAD=zeros(1,dispmax);
depth=zeros(size(left_in));

reverseStr = '';
Crosscheck=1;

for y=wd2+1:1:size(left_res,1)-wd2 %Left image row pix
    %y %Y the scanline
    
    for x=wd2+1:1:size(left_res,2)-wd2  %Left image col pix
        left=left_res(y-wd2:y+wd2,x-wd2:x+wd2); %Left image window w*w
       max_move_left=0;
        for x_disp=dispmin:1:dispmax
           if x-wd2-x_disp<=0 %ignore black region on the left, right bound is constrainted by x
               break
           else
              right=right_res(y-wd2:y+wd2,x-wd2-x_disp:x+wd2-x_disp);
              SAD(x_disp)=sum(abs(left(:)-right(:))); %Take the sum of absolute difference
              %SSD(disp)=SAD.^2;
              max_move_left=x_disp; %valid movement to the left without corssing the borader
           end
           
        end
        if max_move_left~=0
            [temp,depthL]=min(SAD(dispmin:max_move_left));
            depthL=depthL+dispmin-1;
        
            switch Crosscheck
                case 1
            
             if (x-wd2-depthL>0 & x+wd2-depthL<size(left_res,2) & depthL~=0)
                 max_move_right=0;
                 right_rc=right_res(y-wd2:y+wd2,x-wd2-depthL:x+wd2-depthL);         
                 for disp_rc=dispmin:1:dispmax
                     if x+wd2-depthL+disp_rc<size(left_res,2)-wd2 %ignore black region on the left, right bound is constrainted by x
                        left_rc=left_res(y-wd2:y+wd2,x-wd2-depthL+disp_rc:x+wd2-depthL+disp_rc);
                        SAD_rc(disp_rc)=sum(abs(right_rc(:)-left_rc(:))); %Take the sum of absolute difference
                        %SSD(disp)=SAD.^2;
                        max_move_right=disp_rc;
                     end
                 end
                [temp,depth_rc]=min(SAD_rc(dispmin:max_move_right));         
                depth_rc=depth_rc+dispmin-1;
                if abs(depth_rc-depthL)>=disp_th
                   depth(y-wd2,x-wd2)=min(depthL,depth_rc);
                   %depth=min(depthL,depth_rc);
                end

             end
                case 2
                    depth(y-wd2,x-wd2)=depthL;
            end
        end
        
    end
    
   percentDone = 100*y/(size(left_res,1)-wd2);
   msg = sprintf('Percent done: %3.1f', percentDone); %Don't forget this semicolon
   fprintf([reverseStr, msg]);
   reverseStr = repmat(sprintf('\b'), 1, length(msg));
end


figure
imshow(depth,disparityRange)

figure
imagesc(depth)
% 
% 
% 
xyzPoints = reconstructScene(depth,cameracalib.calibrationSession.CameraParameters);
points3D = xyzPoints ./ 1000;
ptCloud = pointCloud(points3D,'Color',leftimg_col);
% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-1,1], [0, 5], 'VerticalAxis', 'y', 'VerticalAxisDir', 'Down');
% Visualize the point cloud
view(player3D, ptCloud);

%% a