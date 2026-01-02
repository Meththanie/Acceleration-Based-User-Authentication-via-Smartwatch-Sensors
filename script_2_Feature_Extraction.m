%  SCRIPT 2 — FEATURE EXTRACTION
%  Loads segments from /Preprocessed
%  Extracts time + frequency features using 6 IMU channels
%  Saves features into /Features
%  Includes REQUIRED VISUALIZATIONS:
%     (1) Example segment (X,Y,Z)
%     (2) Boxplots of features
%     (3) Mean feature values
%  ================================================================
clc; clear; close all;

%% ---------------- PATHS ----------------
PRE_DIR = fullfile(pwd, 'Preprocessed');
OUT_DIR = fullfile(pwd, 'Features');

if ~exist(OUT_DIR,'dir')
    mkdir(OUT_DIR);
end

%% ---------------- SETTINGS ----------------
mode = 'combined';      % 'time' | 'freq' | 'combined'
fs   = 32;               % sampling rate after interpolation
visualize_user = true;  % show graphs for FIRST file only

fprintf('\n=== FEATURE EXTRACTION STARTED ===\n');

%% ---------------- LOAD PREPROCESSED FILES ----------------
files = dir(fullfile(PRE_DIR, '*_segments.mat'));
if isempty(files)
    error('No preprocessed files found in %s', PRE_DIR);
end

fprintf('Found %d preprocessed files.\n\n', length(files));

%% ---------------- PROCESS EACH FILE ----------------
for f = 1:length(files)

    fname = files(f).name;
    fprintf('[%d/%d] Processing %s\n', f, length(files), fname);

    load(fullfile(PRE_DIR, fname), 'segments');

    if isempty(segments)
        warning('  No segments in %s\n', fname);
        continue;
    end

    all_features = [];

    %% ---- Extract features for each segment ----
    for s = 1:length(segments)
        seg = segments{s};

        % seg is size: win_samples × 6  (AccXYZ + GyroXYZ)
        seg_use = seg;   % 6 channels

        fv = extract_features(seg_use, fs);   % returns ~105 features
        all_features = [all_features; fv];
    end

    fprintf('  Extracted %d features per segment.\n', size(all_features,2));

    %% ---- Save features ----
    fshort = fname(1:end-13); % remove "_segments.mat"
    matname = fullfile(OUT_DIR, [fshort '_features.mat']);
    csvname = fullfile(OUT_DIR, [fshort '_features.csv']);

    save(matname, 'all_features');
    writematrix(all_features, csvname);

    fprintf('  Saved: %s\n\n', matname);

    %% ---------------- VISUALIZATION (ONLY FOR FIRST FILE) ----------------
    if visualize_user && f == 1
        fprintf('  Creating visualization...\n');

        figure('Name','Feature Visualization','Position',[50 50 1200 500]);

        % -------- (1) Boxplot of first 10 features --------
        subplot(1,2,1);
        boxplot(all_features(:,1:10));
        title('Boxplot of First 10 Features');
        xlabel('Feature Index'); ylabel('Value');
        grid on;

        % -------- (2) Mean of all features --------
        subplot(1,2,2);
        plot(mean(all_features,1),'LineWidth',1.5);
        title('Mean of All Features');
        xlabel('Feature Index'); ylabel('Mean Value');
        grid on;

        saveas(gcf, fullfile(OUT_DIR,'feature_visualization.png'));
    end

end

fprintf('\n=== FEATURE EXTRACTION COMPLETE ===\n');
fprintf('Features saved in: %s\n', OUT_DIR);

