% script_1_Preprocessing.m
% Segments raw CSV files and interpolates to 150 samples

clc; clear; close all;

DATA_DIR = 'Dataset';
OUT_DIR = 'Preprocessed';
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

fs_orig = 32;
target_fs = 32;
win_sec = 5;
overlap = 0.5;
PLOT_SAMPLE = true;

files = dir(fullfile(DATA_DIR,'U*_*.csv'));
if isempty(files)
    error('No CSV files found in %s', DATA_DIR);
end

fprintf('Found %d files\n', length(files));

for f = 1:length(files)
    
    fname = files(f).name;
    fprintf('\n[%d/%d] %s\n', f, length(files), fname);
    
    % load raw data
    raw = readmatrix(fullfile(DATA_DIR, fname));
    
    if isempty(raw)
        warning('Empty file: %s', fname);
        continue;
    end
    
    % keep first 7 columns only
    if size(raw,2) < 7
        warning('File has fewer than 7 columns, skipping');
        continue;
    end
    raw = raw(:,1:7);
    
    % median filter on sensor data (columns 2-7)
    raw(:,2:7) = medfilt1(raw(:,2:7), 3);
    
    % time vector
    N = size(raw,1);
    t_full = (0:N-1)' / fs_orig;
    
    % segmentation
    win_samples = round(win_sec * target_fs);
    step_samples = max(1, round(win_samples * (1-overlap)));
    
    starts = 1:step_samples:(N - win_samples + 1);
    segments = {};
    
    for i = 1:length(starts)
        s = starts(i);
        e = s + win_samples - 1;
        if e > N, break; end
        
        % get window (only sensor columns 2-7)
        window = raw(s:e, 2:7);
        
        % interpolate to exact length
        x_orig = (0:size(window,1)-1)';
        x_target = linspace(0, size(window,1)-1, win_samples)';
        
        seg_interp = zeros(win_samples, 6);
        
        for c = 1:6
            y = window(:,c);
            [xu, ia] = unique(x_orig);
            yu = y(ia);
            
            if length(xu) < 2
                seg_interp(:,c) = yu(1);
            else
                seg_interp(:,c) = interp1(xu, yu, x_target, 'linear', 'extrap');
            end
        end
        
        segments{end+1} = seg_interp;
    end
    
    % save
    outname = fullfile(OUT_DIR, [fname(1:end-4) '_segments.mat']);
    save(outname, 'segments', 'win_sec', 'target_fs', 'overlap');
    
    fprintf('  Saved %d segments\n', length(segments));
    
    % visualization for first file only
    if PLOT_SAMPLE && f == 1
        figure('Position',[100 100 1000 600]);
        
        % raw snippet
        T = min(N, 4*fs_orig);
        subplot(3,1,1);
        plot(t_full(1:T), raw(1:T,2:4));
        title('Raw Accelerometer Data');
        xlabel('Time (s)'); ylabel('Acceleration');
        legend('X','Y','Z'); grid on;
        
        % with segment markers
        subplot(3,1,2);
        plot(t_full(1:T), raw(1:T,2:4)); hold on;
        for b = starts(starts <= T)
            xline((b-1)/fs_orig, 'k--', 'LineWidth', 0.5);
        end
        title('Segmentation Windows');
        xlabel('Time (s)'); ylabel('Acceleration');
        legend('X','Y','Z'); grid on;
        
        % first interpolated segment
        subplot(3,1,3);
        seg1 = segments{1};
        tx = (0:size(seg1,1)-1)/target_fs;
        plot(tx, seg1(:,1:3));
        title(sprintf('First Segment (%d samples)', size(seg1,1)));
        xlabel('Time (s)'); ylabel('Value');
        legend('X','Y','Z'); grid on;
        
        saveas(gcf, fullfile(OUT_DIR,'preprocessing_sample.png'));
    end
end

fprintf('\nPreprocessing done. Saved in %s\n', OUT_DIR);