function camera = sm_open(device,resolution)
arguments
    device (1,1) string = "ZED"
    resolution (1,1) string = "2560x720"
end
camera = webcam(device);
if ~any(strcmp(camera.AvailableResolutions,resolution))
    error('sm:resolution','Requested side-by-side resolution is unavailable.');
end
camera.Resolution = char(resolution);
end