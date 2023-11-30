function [stateInfo, speed] = run_tracker(curSequence, baselinedetections)
%% tracker configuration
%% EB
sigma_l = 0.4;
sigma_iou = 0.3;
sigma_p = 24;
sigma_len = 3;
skipframes = 0;
skip_factor = 3;

%% running tracking algorithm
try
    ret = py.iou_tracker.track_iou_matlab_wrapper(py.numpy.array(baselinedetections(:).'), sigma_l, sigma_iou, sigma_p, sigma_len, skipframes, skip_factor);
    
catch exception
    disp('error while calling the python tracking module: ')
    disp(' ')
    disp(getReport(exception))
end
speed = ret{1};
track_result = cell2mat(reshape(ret{2}.cell.', 6, []).');

%% convert and save the mot style track_result
stateInfo = saveStateInfo(track_result, numel(curSequence.frameNums));
