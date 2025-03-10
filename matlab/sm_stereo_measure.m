function result = sm_stereo_measure(left,right,stereoParams,options)
arguments
    left
    right
    stereoParams
    options.LeftXY (:,2) double = zeros(0,2)
    options.RightXY (:,2) double = zeros(0,2)
    options.PlyFile (1,1) string = ""
end
if size(options.LeftXY,1) ~= size(options.RightXY,1)
    error('sm:points','Left and right correspondence counts differ.');
end
[leftRect,rightRect,reprojection] = rectifyStereoImages(left,right,stereoParams,'OutputView','valid');
leftGray = leftRect;
rightGray = rightRect;
if size(leftGray,3) == 3
    leftGray = rgb2gray(leftGray);
    rightGray = rgb2gray(rightGray);
end
disparityMap = disparitySGM(leftGray,rightGray,'DisparityRange',[0 128],'UniquenessThreshold',15);
xyz = reconstructScene(disparityMap,reprojection);
valid = all(isfinite(xyz),3) & disparityMap > 0 & xyz(:,:,3) > 0 & xyz(:,:,3) < 10000;
points = reshape(xyz,[],3);
cloud = pointCloud(points(valid(:),:) / 1000);
result = struct('Disparity',disparityMap,'Valid',valid,'CloudMeters',cloud);
if ~isempty(options.LeftXY)
    leftPoints = undistortPoints(options.LeftXY,stereoParams.CameraParameters1);
    rightPoints = undistortPoints(options.RightXY,stereoParams.CameraParameters2);
    [worldPoints,errors] = triangulate(leftPoints,rightPoints,stereoParams);
    result.WorldPointsMm = worldPoints;
    result.ReprojectionErrorsPx = errors;
    result.EpipolarLines = epipolarLine(stereoParams.FundamentalMatrix,leftPoints);
    if size(worldPoints,1) >= 2 && all(errors(1:2) < 2) && all(worldPoints(1:2,3) > 0)
        result.DistanceMm = norm(worldPoints(1,:)-worldPoints(2,:));
    end
end
figure('Name','Stereo measurements'); tiledlayout(2,2);
nexttile; imshowpair(left,right,'ColorChannels','red-cyan'); title('Raw anaglyph');
nexttile; imshowpair(leftRect,rightRect,'ColorChannels','red-cyan'); title('Rectified anaglyph');
nexttile; imshow(disparityMap,[0 128]); colorbar; title('Disparity');
nexttile; pcshow(cloud); title('Point cloud [meters]');
if strlength(options.PlyFile) > 0
    pcwrite(cloud,options.PlyFile,'Encoding','ascii');
end
end