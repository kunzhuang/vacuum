%RMS error
function [RMS,RMSAll, perCorr,perCorrAll]=RMSErr(RefImg,Unreliable,corr_th)
ocol_val=1;

TrueImg=imread('cond.png');
RefImg=RefImg(:,ocol_val:end);

TrueImgAll=double(TrueImg);
RefImgAll=double(RefImg);

if ~exist('Unreliable','var') || isempty(Unreliable)
    TrueImg=TrueImg(:,ocol_val:end);
else
    TrueImg=double(TrueImg).*Unreliable;
    RefImg=double(RefImg).*Unreliable;
    TrueImg=TrueImg(:,ocol_val:end);
end



TrueImg=double(TrueImg)/3; TrueImgAll=double(TrueImgAll)/3;
RefImg=double(RefImg);

RMS=sqrt(sum(sum((abs(TrueImg-RefImg)).^2))/(size(TrueImg,1)*(size(TrueImg,2))));

RMSAll=sqrt(sum(sum((abs(TrueImgAll-RefImgAll)).^2))/(size(TrueImgAll,1)*(size(TrueImgAll,2))));
%SSIM=ssim(RefImg,TrueImg);

perCorr=sum(sum(abs(TrueImg-RefImg)<=corr_th))/(size(RefImg,1)*size(RefImg,2));
perCorrAll=sum(sum(abs(TrueImgAll-RefImgAll)<=corr_th))/(size(TrueImgAll,1)*size(TrueImgAll,2));


diffs=TrueImg-RefImg;
temp=(abs(diffs))/max(max(abs(diffs)));

figure
imshow(uint8(255*(temp)));
colorbar

figure
TrueImgAll=double(TrueImgAll);
imagesc(255*TrueImgAll/max(max(TrueImgAll)))
colormap jet
colorbar
end