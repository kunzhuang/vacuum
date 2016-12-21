function [CKLR,PixelLeft]= LRRLConsistency(LeftDisp,RightDisp,dis_th)
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
PixelLeft=sum(sum(CKLR))/(Cols*Rows);
end