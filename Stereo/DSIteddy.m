focal=1331;
yline1=60;
yline2=190;
yline3=240;
base=0.2;

DSI=PixelCostL;

slice1 = squeeze(PixelCostL(yline1,:,:));
slice2= squeeze(PixelCostL(yline2,:,:));
slice3 = squeeze(PixelCostL(yline3,:,:));

sliceT=imread('d.png')/4;
sliceT1=sliceT(yline1,:);
sliceT2=sliceT(yline2,:);
sliceT3=sliceT(yline3,:);

figure
subplot(6,1,1);
imagesc(slice1');
ylabel('D-axis');xlabel('X-axis')
set(gca,'Ydir','normal');
title('(a) DSI slice at y=60') 

subplot(6,1,2);
plot(sliceT1)
ylabel('D-axis');xlabel('X-axis')
axis([0,450,0,64])
title('(b) Ground truth at y=60') 


subplot(6,1,3);
imagesc(slice2');
ylabel('D-axis');xlabel('X-axis')
set(gca,'Ydir','normal');
title('(c) DSI slice at y=190') 

subplot(6,1,4);
plot(sliceT2)
ylabel('D-axis');xlabel('X-axis')
axis([0,450,0,64])
title('(d) Ground truth at y=190') 

subplot(6,1,5);
imagesc(slice3');
ylabel('D-axis');xlabel('X-axis')
set(gca,'Ydir','normal');
title('(e) DSI slice at y=240') 

subplot(6,1,6);
plot(sliceT3)
ylabel('D-axis');xlabel('X-axis')
axis([0,450,0,64])
title('(f) Ground truth at y=240') 

