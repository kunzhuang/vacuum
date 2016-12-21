function disparityMap=ComputeDisparity2(Rows, Cols, disparityRange,AggregatedCostArrary)
for Col=1:Cols
    for Row=1:Rows
        [~,d]=min(AggregatedCostArrary(Row,Col,:));
        disparity = d+disparityRange(1,1)-1;
        disparityMap(Row,Col)=disparity;
    end
end
end