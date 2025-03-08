function [left,right] = sm_grab(camera)
frame = snapshot(camera);
width = size(frame,2);
if mod(width,2) ~= 0
    error('sm:width','Side-by-side frame width must be even.');
end
left = frame(:,1:width/2,:);
right = frame(:,width/2+1:end,:);
end