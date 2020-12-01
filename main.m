if(~exist('net'))
    net = load('netLSTM.mat');
    netCNN = net.netCNN;
    netLSTM = net.netLSTM;
    inputSize = net.inputSize;
    layerName = net.layerName;
    label = net.label; 
end

% Connect to webcam
wcam = webcam(1);
wcam.Resolution = wcam.AvailableResolutions{end};
player = vision.DeployableVideoPlayer();

if exist('file.mp4.avi', 'file')==2
  delete('file.mp4.avi');
end

f = VideoWriter('file.mp4','MOtion JPEG AVI');
open(f);
cont= true;
disp('Start Signing for 2 sec')
for iFrame=1:60
    img = snapshot(wcam);
    tic; % Count FPS
    writeVideo(f,img);
end
release(player);
close(f);
disp('Done recording')
v = 'file.mp4.avi';
video=readVideo(v);
video = centerCrop(video,inputSize);
testSet = activations(netCNN,video,layerName,'OutputAs','columns');
v=netLSTM.predict(testSet);
netLSTM.classify(testSet)
clear wcam;
implay('file.mp4.avi')
release(player);


function videoResized = centerCrop(video,inputSize)
sz = size(video);
if sz(1) < sz(2)
    % Video is landscape
    idx = floor((sz(2) - sz(1))/2);
    video(:,1:(idx-1),:,:) = [];
    video(:,(sz(1)+1):end,:,:) = [];
    
elseif sz(2) < sz(1)
    % Video is portrait
    idx = floor((sz(1) - sz(2))/2);
    video(1:(idx-1),:,:,:) = [];
    video((sz(2)+1):end,:,:,:) = [];
end
 
videoResized = imresize(video,inputSize(1:2));
 end
 
 function video = readVideo(filename)
vr = VideoReader(filename);
H = vr.Height;
W = vr.Width;
C = 3;
% Preallocate video array
numFrames = floor(vr.Duration * vr.FrameRate);
video = zeros(H,W,C,numFrames);
% Read frames
i = 0;
while hasFrame(vr)
    i = i + 1;
    video(:,:,:,i) = readFrame(vr);
end
% Remove unallocated frames
if size(video,4) > i
    video(:,:,:,i+1:end) = [];
end
 end