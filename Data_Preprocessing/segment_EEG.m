% this function will output n EEG epochs corresponding to n EEG events
% Each epoch lasts 0.5s (0.5*sampling_rate data points) 
function generate(save_root, txt, samples, down_ratio, n, video)
    global data;
    global trigger;
    data = data(1:96, :)';
    channel = 96;
    EEG = zeros(n, samples, channel);
    for i = 1:length(trigger)
        % the samples is 4096*0.5 = 0.5*sampling_rate => equal to 0.5s
        EEG(i, :, :) = reshape(data(trigger(i):trigger(i)+samples-1, :), [1, samples, channel]);
    end
    EEG = reshape(EEG, [n*samples, channel]);
    ave = mean(EEG, 1);
    STD = std(EEG, 1);
    for i = 1:channel
       EEG(:, i) = (EEG(:, i)-ave(i))/STD(i);
    end
    EEG = reshape(EEG, [n, samples, channel]);
    fid = fopen(txt);
    tline = fgetl(fid);
    i = 1;
    while ischar(tline)
        if video==0
            tline = tline(1:end-5); %ignore '.jpeg'
        end
        tline = [tline, '.mat'];
        eeg = reshape(EEG(i, :, :), [samples, channel]);
        eeg = eeg(1:down_ratio:end, :)';
        name = [save_root, tline];
        save(name, 'eeg');
        tline = fgetl(fid);
        i = i+1;
    end
    