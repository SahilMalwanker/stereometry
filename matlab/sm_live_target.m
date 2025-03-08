function sm_live_target(camera,frameCount)
arguments
    camera
    frameCount (1,1) double {mustBePositive,mustBeInteger} = 100
end
viewer = figure('Name','Checkerboard detection');
for index = 1:frameCount
    if ~isvalid(viewer)
        break
    end
    [left,~] = sm_grab(camera);
    [points,~] = detectCheckerboardPoints(left);
    imshow(left); hold on;
    if ~isempty(points)
        plot(points(:,1),points(:,2),'r+');
    end
    hold off; drawnow;
end
end