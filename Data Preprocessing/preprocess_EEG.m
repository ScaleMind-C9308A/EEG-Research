addpath('D:\Data\CVPR2021-02785\CVPR2021-02785\code\TeamX_code\');
addpath('D:\Download\eeglab2020_0\eeglab2020_0');

src_path = 'D:\Data\CVPR2021-02785\CVPR2021-02785\data\imagenet40-1000-';
preprocessed_path = 'D:\Data\CVPR2021-02785\CVPR2021-02785\preprocessed\imagenet40-1000-';
design_path = 'D:\Data\CVPR2021-02785\CVPR2021-02785\design\run';

eeglab;

band = 0;
notch = 0;

sampling_rate = 4096;
down_factor = 4;

for subject = 1:1
    for run = 0:0
        bdf = sprintf('%s%d-%02d.bdf', src_path, subject, run);
        out = sprintf('%s%d/', preprocessed_path, subject);
        stim = sprintf('%s-%02d.txt', design_path, run);
        trim = run==14;
        read_EEG(bdf, band, notch, 400, trim);
        segment_EEG(out, stim, sampling_rate*0.5, down_factor, 400, 0);
    end
end

exit