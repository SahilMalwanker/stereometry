function saved = sm_capture_sequence(camera,folder,count)
arguments
    camera
    folder (1,1) string
    count (1,1) double {mustBePositive,mustBeInteger} = 20
end
leftFolder = fullfile(folder,'left');
rightFolder = fullfile(folder,'right');
if ~isfolder(leftFolder), mkdir(leftFolder); end
if ~isfolder(rightFolder), mkdir(rightFolder); end
saved = 0;
while saved < count
    reply = input('Reposition the board; Enter to capture, q to stop: ','s');
    if strcmpi(reply,'q'), break; end
    [left,right] = sm_grab(camera);
    [leftPoints,leftSize] = detectCheckerboardPoints(left);
    [rightPoints,rightSize] = detectCheckerboardPoints(right);
    if isempty(leftPoints) || isempty(rightPoints) || ~isequal(leftSize,rightSize)
        disp('Board not detected consistently in both views; pair skipped.');
        continue
    end
    filename = sprintf('pair_%03d.png',saved+1);
    leftPath = fullfile(leftFolder,filename);
    rightPath = fullfile(rightFolder,filename);
    if isfile(leftPath) || isfile(rightPath)
        error('sm:exists','Output pair already exists. Choose a new directory.');
    end
    imwrite(left,leftPath);
    imwrite(right,rightPath);
    saved = saved+1;
end
end