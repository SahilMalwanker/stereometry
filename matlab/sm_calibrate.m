function [mono,stereo] = sm_calibrate(folder,squareMm)
arguments
    folder (1,1) string
    squareMm (1,1) double {mustBePositive}
end
leftFiles = dir(fullfile(folder,'left','*.png'));
rightFiles = dir(fullfile(folder,'right','*.png'));
if numel(leftFiles) < 6 || ~isequal(sort({leftFiles.name}),sort({rightFiles.name}))
    error('sm:pairs','Need at least six identically named left/right PNG pairs.');
end
[~,order] = sort({leftFiles.name});
leftFiles = leftFiles(order);
names = string({leftFiles.name});
leftPaths = fullfile(folder,'left',names);
rightPaths = fullfile(folder,'right',names);
sample = imread(leftPaths(1));
imageSize = [size(sample,1),size(sample,2)];
[pairedPoints,boardSize] = detectCheckerboardPoints(cellstr(leftPaths),cellstr(rightPaths));
world = generateCheckerboardPoints(boardSize,squareMm);
stereo = estimateCameraParameters(pairedPoints,world,'ImageSize',imageSize,...
    'NumRadialDistortionCoefficients',3,'EstimateTangentialDistortion',true,'EstimateSkew',false);
[monoPoints,monoBoardSize] = detectCheckerboardPoints(cellstr(leftPaths));
monoWorld = generateCheckerboardPoints(monoBoardSize,squareMm);
mono = estimateCameraParameters(monoPoints,monoWorld,'ImageSize',imageSize,...
    'NumRadialDistortionCoefficients',3,'EstimateTangentialDistortion',true,'EstimateSkew',false);
figure; showReprojectionErrors(stereo);
figure; showExtrinsics(stereo);
save(fullfile(folder,'calibration.mat'),'mono','stereo','squareMm');
end