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