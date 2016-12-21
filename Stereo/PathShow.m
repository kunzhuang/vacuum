for p=1:16
    figure
    filename = [ 'Path1_', num2str(p), '.mat' ];
    PathImshow(p)=load(filename);
    %imshow(255*path(p).Path/max(max(path(p).Path)),[1,255])
    imshow(255*PathImshow(p).PathIMG/max(max(PathImshow(p).PathIMG)),[1,255])
end

