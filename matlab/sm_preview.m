function sm_preview(camera)
[left,right] = sm_grab(camera);
figure('Name','Stereo acquisition');
imshowpair(left,right,'montage');
title('Left and right from one side-by-side frame');
end