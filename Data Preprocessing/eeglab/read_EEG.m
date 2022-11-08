function read_EEG(filename, band, notch, n, trim)
    global data;
    global trigger;
    EEG = pop_biosig(filename);
    EEG = pop_reref(EEG, [97 98]);
    if band==1
        EEG = pop_eegfiltnew(EEG, [], 14, 3862, 1, [], 0);
        EEG = pop_eegfiltnew(EEG, [], 71, 762, 0, [], 0);
    end
    if notch==1
        EEG = pop_eegfiltnew(EEG, 49, 51, 13518, 1, [], 0);
    end
    data = EEG.data(1:96, :);
    trigger = EEG.event; % Assign eeg events to trigget variable
    trigger = struct2cell(trigger); % Convert struct to cell array
    trigger = cell2mat(trigger); % Convert cell array to matlab array
    trigger = trigger(2, :, :);
    trigger = trigger(2:end);
    if trim==1
        trigger = trigger(:, :, 1:n);
    end
    trigger = reshape(trigger, [1, n]);
    