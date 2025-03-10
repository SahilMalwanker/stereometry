function result = sm_mono_measure(image,cameraParams,squareMm)
arguments
    image
    cameraParams
    squareMm (1,1) double {mustBePositive}
end
corrected = undistortImage(image,cameraParams,'OutputView','same');
[boardPoints,boardSize] = detectCheckerboardPoints(corrected);
if isempty(boardPoints)
    error('sm:board','No checkerboard detected in the corrected image.');
end
world = generateCheckerboardPoints(boardSize,squareMm);
intrinsics = cameraIntrinsics(cameraParams.FocalLength,cameraParams.PrincipalPoint,cameraParams.ImageSize);
[rotation,translation] = extrinsics(boardPoints,world,intrinsics);
figure; imshow(corrected); title('Select two points in the board plane');
[xCoordinates,yCoordinates] = ginput(2);
if numel(xCoordinates) ~= 2
    error('sm:points','Exactly two points are required.');
end
worldXY = pointsToWorld(intrinsics,rotation,translation,[xCoordinates,yCoordinates]);
result = struct('WorldXY',worldXY,'DistanceMm',norm(worldXY(1,:)-worldXY(2,:)));
disp(result);
end