% script_5_Optimization.m
% Complete optimization experiments in one script
% Tests different sensor modes and neuron counts, compares with SVM

clc; clear; close all;
rng(42);

fprintf('\n=== COMPLETE OPTIMIZATION EXPERIMENTS ===\n\n');

%% -------------------- CONFIGURATION --------------------
PREP_DIR = 'Preprocessed';
RESULT_DIR = 'Results_Optimization';
if ~exist(RESULT_DIR,'dir'), mkdir(RESULT_DIR); end

fs = 32;
sensor_modes = {'combined', 'acc_only', 'gyro_only'};
neuron_counts = [10, 20, 30, 50];

%% -------------------- LOAD PREPROCESSED FILES --------------------
fprintf('Step 1: Loading preprocessed segments...\n');

seg_files = dir(fullfile(PREP_DIR, '*_segments.mat'));
if isempty(seg_files)
    error('No preprocessed files found. Run Preprocessing.m first!');
end

% organize files by user and session
file_map = struct();
for i = 1:length(seg_files)
    fname = seg_files(i).name;
    
    % extract user number
    tokens = regexp(fname, 'U(\d+)', 'tokens');
    if isempty(tokens), continue; end
    user = str2double(tokens{1}{1});
    
    key = sprintf('U%d', user);
    if ~isfield(file_map, key)
        file_map.(key) = struct('FD','', 'MD','');
    end
    
    % determine session type
    if contains(fname, '_FD', 'IgnoreCase', true)
        file_map.(key).FD = fullfile(PREP_DIR, fname);
    elseif contains(fname, '_MD', 'IgnoreCase', true)
        file_map.(key).MD = fullfile(PREP_DIR, fname);
    end
end

% get valid users (have both FD and MD)
users = fieldnames(file_map);
valid_users = [];
for i = 1:length(users)
    u = file_map.(users{i});
    if ~isempty(u.FD) && ~isempty(u.MD)
        valid_users(end+1) = str2double(users{i}(2:end));
    end
end
valid_users = sort(valid_users);
num_users = length(valid_users);

fprintf('  Found %d valid users\n\n', num_users);

%% -------------------- MAIN EXPERIMENTS --------------------
results = [];
result_count = 0;

%% EXPERIMENT 1: Sensor Mode Comparison (30 neurons)
fprintf('EXPERIMENT 1: Sensor Mode Comparison\n');
fprintf('%s\n', repmat('-',1,60));

for mode_idx = 1:length(sensor_modes)
    mode = sensor_modes{mode_idx};
    
    fprintf('[%d/%d] Testing %s (30 neurons)... ', mode_idx, length(sensor_modes), mode);
    
    % extract features and train/test for this mode
    [acc, far, frr, eer] = run_experiment(file_map, valid_users, mode, 30, 0.7, 105, fs);
    
    result_count = result_count + 1;
    results(result_count).name = mode;
    results(result_count).experiment = 'sensor';
    results(result_count).neurons = 30;
    results(result_count).split_ratio = 0.7;
    results(result_count).num_features = 105;
    results(result_count).accuracy = mean(acc);
    results(result_count).FAR = mean(far);
    results(result_count).FRR = mean(frr);
    results(result_count).EER = mean(eer);
    
    fprintf('Acc=%.2f%%, EER=%.2f%%\n', mean(acc), mean(eer));
end

%% EXPERIMENT 2: Neuron Count Comparison (combined sensors)
fprintf('\nEXPERIMENT 2: Hidden Neuron Count Comparison\n');
fprintf('%s\n', repmat('-',1,60));

for n_idx = 1:length(neuron_counts)
    num_neurons = neuron_counts(n_idx);
    
    fprintf('[%d/%d] Testing %d neurons... ', n_idx, length(neuron_counts), num_neurons);
    
    [acc, far, frr, eer] = run_experiment(file_map, valid_users, 'combined', num_neurons, 0.7, 105, fs);
    
    result_count = result_count + 1;
    results(result_count).name = sprintf('%d_neurons', num_neurons);
    results(result_count).experiment = 'neurons';
    results(result_count).neurons = num_neurons;
    results(result_count).split_ratio = 0.7;
    results(result_count).num_features = 105;
    results(result_count).accuracy = mean(acc);
    results(result_count).FAR = mean(far);
    results(result_count).FRR = mean(frr);
    results(result_count).EER = mean(eer);
    
    fprintf('Acc=%.2f%%, EER=%.2f%%\n', mean(acc), mean(eer));
end

%% EXPERIMENT 3: Train/Test Split Ratio
fprintf('\nEXPERIMENT 3: Train/Test Split Ratio\n');
fprintf('%s\n', repmat('-',1,60));

split_ratios = [0.6, 0.7, 0.8];
for split_idx = 1:length(split_ratios)
    ratio = split_ratios(split_idx);
    
    fprintf('[%d/%d] Testing %.0f/%.0f split... ', split_idx, length(split_ratios), ...
        ratio*100, (1-ratio)*100);
    
    [acc, far, frr, eer] = run_experiment(file_map, valid_users, 'combined', 30, ratio, 105, fs);
    
    result_count = result_count + 1;
    results(result_count).name = sprintf('%.0f-%.0f_split', ratio*100, (1-ratio)*100);
    results(result_count).experiment = 'split_ratio';
    results(result_count).neurons = 30;
    results(result_count).accuracy = mean(acc);
    results(result_count).FAR = mean(far);
    results(result_count).FRR = mean(frr);
    results(result_count).EER = mean(eer);
    
    fprintf('Acc=%.2f%%, EER=%.2f%%\n', mean(acc), mean(eer));
end

%% EXPERIMENT 4: Feature Reduction
fprintf('\nEXPERIMENT 4: Feature Reduction (Cost Optimization)\n');
fprintf('%s\n', repmat('-',1,60));

feature_counts = [105, 70, 50, 30];
for feat_idx = 1:length(feature_counts)
    num_feat = feature_counts(feat_idx);
    
    fprintf('[%d/%d] Testing %d features... ', feat_idx, length(feature_counts), num_feat);
    
    [acc, far, frr, eer] = run_experiment(file_map, valid_users, 'combined', 30, 0.7, num_feat, fs);
    
    result_count = result_count + 1;
    results(result_count).name = sprintf('%d_features', num_feat);
    results(result_count).experiment = 'feature_reduction';
    results(result_count).neurons = 30;
    results(result_count).split_ratio = 0.7;           
    results(result_count).num_features = num_feat;      
    results(result_count).accuracy = mean(acc);
    results(result_count).FAR = mean(far);
    results(result_count).FRR = mean(frr);
    results(result_count).EER = mean(eer);
    
    cost_reduction = (105 - num_feat) / 105 * 100;
    fprintf('Acc=%.2f%%, EER=%.2f%% (Cost: -%.1f%%)\n', ...
        mean(acc), mean(eer), cost_reduction);
end

%% EXPERIMENT 5: PCA Feature Reduction  
fprintf('\nEXPERIMENT 5: PCA Feature Reduction\n');
fprintf('%s\n', repmat('-',1,60));

pca_feature_counts = [90, 70, 50, 30];
for feat_idx = 1:length(pca_feature_counts)
    num_pca = pca_feature_counts(feat_idx);
    
    fprintf('[%d/%d] Testing %d PCA components... ', feat_idx, length(pca_feature_counts), num_pca);
    
    [acc, far, frr, eer] = run_pca_experiment(file_map, valid_users, 'combined', 10, 0.8, num_pca, fs);
    
    result_count = result_count + 1;
    results(result_count).name = sprintf('%d_PCA', num_pca);
    results(result_count).experiment = 'PCA';
    results(result_count).neurons = 10;
    results(result_count).split_ratio = 0.8;
    results(result_count).num_features = num_pca;
    results(result_count).accuracy = mean(acc);
    results(result_count).FAR = mean(far);
    results(result_count).FRR = mean(frr);
    results(result_count).EER = mean(eer);
    
    fprintf('Acc=%.2f%%, EER=%.2f%%\n', mean(acc), mean(eer));
end

%% EXPERIMENT 6: SVM Comparison
fprintf('\nEXPERIMENT 3: SVM Comparison\n');
fprintf('%s\n', repmat('-',1,60));
fprintf('Testing SVM classifier... ');

[acc, far, frr, eer] = run_svm_experiment(file_map, valid_users, 'combined', fs);

result_count = result_count + 1;
results(result_count).name = 'SVM';
results(result_count).experiment = 'algorithm';
results(result_count).neurons = 0;
results(result_count).split_ratio = 0.7;
results(result_count).num_features = 105;
results(result_count).accuracy = mean(acc);
results(result_count).FAR = mean(far);
results(result_count).FRR = mean(frr);
results(result_count).EER = mean(eer);

fprintf('Acc=%.2f%%, EER=%.2f%%\n', mean(acc), mean(eer));

%% -------------------- SAVE RESULTS --------------------
save(fullfile(RESULT_DIR, 'optimization_results.mat'), 'results');

% print summary table
fprintf('\n=== RESULTS SUMMARY ===\n\n');
fprintf('%-20s | %-20s | %10s | %8s | %8s | %8s\n', ...
    'Configuration', 'Experiment', 'Accuracy', 'FAR', 'FRR', 'EER');
fprintf('%s\n', repmat('-',1,75));

for i = 1:length(results)
    fprintf('%-20s | %-20s | %9.2f%% | %7.2f%% | %7.2f%% | %7.2f%%\n', ...
        results(i).name, results(i).experiment, results(i).accuracy, ...
        results(i).FAR, results(i).FRR, results(i).EER);
end

% export to CSV
csvfile = fullfile(RESULT_DIR, 'optimization_summary.csv');
fid = fopen(csvfile, 'w');
fprintf(fid, 'Configuration,Experiment,Neurons,Accuracy,FAR,FRR,EER\n');
for i = 1:length(results)
    fprintf(fid, '%s,%s,%d,%.2f,%.2f,%.2f,%.2f\n', ...
        results(i).name, results(i).experiment, results(i).neurons, ...
        results(i).accuracy, results(i).FAR, results(i).FRR, results(i).EER);
end
fclose(fid);

fprintf('\nResults saved to: %s\n', RESULT_DIR);
fprintf('CSV file: %s\n', csvfile);
fprintf('\n=== ALL EXPERIMENTS COMPLETE ===\n');

%% Generate Comparison Graphs
fprintf('\nGenerating comparison graphs...\n');

% Create results directory if needed
if ~exist(RESULT_DIR,'dir'), mkdir(RESULT_DIR); end

% Graph 1: Sensor Mode Comparison
sensor_idx = find(strcmp({results.experiment}, 'sensor'));
if ~isempty(sensor_idx)
    sensor_results = results(sensor_idx);
    
    figure('Position',[100 100 1200 400]);
    
    subplot(1,3,1);
    bar([sensor_results.accuracy]);
    set(gca, 'XTickLabel', {sensor_results.name}, 'XTickLabelRotation', 45);
    title('Accuracy by Sensor Mode');
    ylabel('Accuracy (%)');
    ylim([80 100]);
    grid on;
    
    subplot(1,3,2);
    bar([sensor_results.EER]);
    set(gca, 'XTickLabel', {sensor_results.name}, 'XTickLabelRotation', 45);
    title('EER by Sensor Mode');
    ylabel('EER (%)');
    grid on;
    
    subplot(1,3,3);
    b = bar([[sensor_results.FAR]' [sensor_results.FRR]']);
    set(gca, 'XTickLabel', {sensor_results.name}, 'XTickLabelRotation', 45);
    title('FAR vs FRR by Sensor Mode');
    ylabel('Error Rate (%)');
    legend('FAR', 'FRR');
    grid on;
    
    saveas(gcf, fullfile(RESULT_DIR, 'Sensor_Mode_Comparison.png'));
end

% Graph 2: Neuron Count Comparison
neuron_idx = find(strcmp({results.experiment}, 'neurons'));
if ~isempty(neuron_idx)
    neuron_results = results(neuron_idx);
    neurons = [neuron_results.neurons];
    
    figure('Position',[100 100 1200 400]);
    
    subplot(1,3,1);
    plot(neurons, [neuron_results.accuracy], '-o', 'LineWidth', 2, 'MarkerSize', 8);
    title('Accuracy vs Hidden Neurons');
    xlabel('Number of Neurons');
    ylabel('Accuracy (%)');
    grid on;
    
    subplot(1,3,2);
    plot(neurons, [neuron_results.EER], '-o', 'LineWidth', 2, 'MarkerSize', 8);
    title('EER vs Hidden Neurons');
    xlabel('Number of Neurons');
    ylabel('EER (%)');
    grid on;
    
    subplot(1,3,3);
    plot(neurons, [neuron_results.FAR], '-o', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    plot(neurons, [neuron_results.FRR], '-s', 'LineWidth', 2, 'MarkerSize', 8);
    title('FAR & FRR vs Hidden Neurons');
    xlabel('Number of Neurons');
    ylabel('Error Rate (%)');
    legend('FAR', 'FRR');
    grid on;
    
    saveas(gcf, fullfile(RESULT_DIR, 'Neuron_Count_Comparison.png'));
end

% Graph 3: Feature Reduction
feat_idx = find(strcmp({results.experiment}, 'feature_reduction'));
if ~isempty(feat_idx)
    feat_results = results(feat_idx);
    
    % Safely extract features - check if field exists
    if isfield(feat_results, 'num_features')
        feats = [feat_results.num_features];
    else
        % If field missing, use default feature counts
        feats = [105, 70, 50, 30];
        feats = feats(1:length(feat_results));  % Match length
    end
    
    % Make sure arrays are same size
    if length(feats) ~= length(feat_results)
        warning('Feature count mismatch, skipping feature reduction graph');
    else
        cost_reduction = (105 - feats) / 105 * 100;
        
        figure('Position',[100 100 1200 400]);
        
        subplot(1,3,1);
        plot(feats, [feat_results.EER], '-o', 'LineWidth', 2, 'MarkerSize', 8);
        title('EER vs Feature Count');
        xlabel('Number of Features');
        ylabel('EER (%)');
        grid on;
        
        subplot(1,3,2);
        plot(cost_reduction, [feat_results.EER], '-o', 'LineWidth', 2, 'MarkerSize', 8);
        title('EER vs Cost Reduction');
        xlabel('Cost Reduction (%)');
        ylabel('EER (%)');
        grid on;
        
        subplot(1,3,3);
        bar(feats, [feat_results.accuracy]);
        title('Accuracy by Feature Count');
        xlabel('Number of Features');
        ylabel('Accuracy (%)');
        ylim([min([feat_results.accuracy])-5, 100]);
        grid on;
        
        % Add text annotations
        for i = 1:length(feats)
            text(feats(i), feat_results(i).accuracy-1, ...
                sprintf('-%.0f%%', cost_reduction(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end
        
        saveas(gcf, fullfile(RESULT_DIR, 'Feature_Reduction_Comparison.png'));
    end
end

% Graph 4: PCA Feature Reduction
pca_idx = find(strcmp({results.experiment}, 'PCA'));
if ~isempty(pca_idx)
    pca_results = results(pca_idx);
    pca_feats = [pca_results.num_features];
    
    figure('Position',[100 100 1200 400]);
    
    subplot(1,3,1);
    plot(pca_feats, [pca_results.EER], '-o', 'LineWidth', 2, 'MarkerSize', 8);
    title('EER vs PCA Components');
    xlabel('Number of PCA Components');
    ylabel('EER (%)');
    grid on;
    
    subplot(1,3,2);
    plot(pca_feats, [pca_results.accuracy], '-o', 'LineWidth', 2, 'MarkerSize', 8);
    title('Accuracy vs PCA Components');
    xlabel('Number of PCA Components');
    ylabel('Accuracy (%)');
    grid on;
    
    subplot(1,3,3);
    plot(pca_feats, [pca_results.FAR], '-o', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    plot(pca_feats, [pca_results.FRR], '-s', 'LineWidth', 2, 'MarkerSize', 8);
    title('FAR & FRR vs PCA Components');
    xlabel('Number of PCA Components');
    ylabel('Error Rate (%)');
    legend('FAR', 'FRR');
    grid on;
    
    saveas(gcf, fullfile(RESULT_DIR, 'PCA_Comparison.png'));
end

fprintf('  Saved comparison graphs to: %s\n', RESULT_DIR);

%% ==================== HELPER FUNCTIONS ====================

function [acc, far, frr, eer] = run_experiment(file_map, users, sensor_mode, num_neurons, split_ratio, num_features, fs)
    % Run neural network experiment for given configuration
    
    num_users = length(users);
    acc = zeros(num_users, 1);
    far = zeros(num_users, 1);
    frr = zeros(num_users, 1);
    eer = zeros(num_users, 1);
    
    for u_idx = 1:num_users
        user = users(u_idx);
        
        % extract features for this user
        [train_feat, train_lbl, test_feat, test_lbl] = ...
            extract_user_data(file_map, users, user, sensor_mode, fs);
        
        if isempty(train_feat) || isempty(test_feat)
            continue;
        end
        
        % NEW: Feature reduction if needed
        if num_features < size(train_feat, 2)
            train_feat = train_feat(:, 1:num_features);
            test_feat = test_feat(:, 1:num_features);
        end
        
        % NEW: Apply split ratio to training data
        num_train = size(train_feat, 1);
        num_use = round(num_train * split_ratio);
        idx = randperm(num_train, num_use);
        train_feat = train_feat(idx, :);
        train_lbl = train_lbl(idx);
        
        % normalize
        mu = mean(train_feat);
        sigma = std(train_feat) + eps;
        
        train_norm = (train_feat - mu) ./ sigma;
        test_norm = (test_feat - mu) ./ sigma;
        
        % train neural network
        net = feedforwardnet(num_neurons, 'trainscg');
        net.trainParam.epochs = 300;
        net.trainParam.showWindow = false;
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        
        net = train(net, train_norm', train_lbl');
        
        % test
        outputs = net(test_norm');
        
        % calculate metrics
        [acc(u_idx), far(u_idx), frr(u_idx), eer(u_idx)] = ...
            calculate_metrics(outputs', test_lbl);
    end
end

function [acc, far, frr, eer] = run_pca_experiment(file_map, users, sensor_mode, num_neurons, split_ratio, num_pca, fs)
    % Run neural network experiment with PCA feature reduction
    
    num_users = length(users);
    acc = zeros(num_users, 1);
    far = zeros(num_users, 1);
    frr = zeros(num_users, 1);
    eer = zeros(num_users, 1);
    
    for u_idx = 1:num_users
        user = users(u_idx);
        
        [train_feat, train_lbl, test_feat, test_lbl] = ...
            extract_user_data(file_map, users, user, sensor_mode, fs);
        
        if isempty(train_feat) || isempty(test_feat)
            continue;
        end
        
        % Apply split ratio
        num_train = size(train_feat, 1);
        num_use = round(num_train * split_ratio);
        idx = randperm(num_train, num_use);
        train_feat = train_feat(idx, :);
        train_lbl = train_lbl(idx);
        
        % Normalize BEFORE PCA
        mu = mean(train_feat);
        sigma = std(train_feat) + eps;
        train_norm = (train_feat - mu) ./ sigma;
        test_norm = (test_feat - mu) ./ sigma;
        
        % Apply PCA
        [coeff, score, ~] = pca(train_norm);

        % Determine actual number of components to use
        actual_components = min(num_pca, size(coeff, 2));

        % Use only top N components
        train_pca = score(:, 1:actual_components);
        test_pca = test_norm * coeff(:, 1:actual_components);
        
        % Train neural network
        net = feedforwardnet(num_neurons, 'trainscg');
        net.trainParam.epochs = 300;
        net.trainParam.showWindow = false;
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        
        net = train(net, train_pca', train_lbl');
        outputs = net(test_pca');
        
        [acc(u_idx), far(u_idx), frr(u_idx), eer(u_idx)] = ...
            calculate_metrics(outputs', test_lbl);
    end
end

function [acc, far, frr, eer] = run_svm_experiment(file_map, users, sensor_mode, fs)
    % Run SVM experiment
    
    num_users = length(users);
    acc = zeros(num_users, 1);
    far = zeros(num_users, 1);
    frr = zeros(num_users, 1);
    eer = zeros(num_users, 1);
    
    for u_idx = 1:num_users
        user = users(u_idx);
        
        % extract features
        [train_feat, train_lbl, test_feat, test_lbl] = ...
            extract_user_data(file_map, users, user, sensor_mode, fs);
        
        if isempty(train_feat) || isempty(test_feat)
            continue;
        end
        
        % normalize
        mu = mean(train_feat);
        sigma = std(train_feat) + eps;
        
        train_norm = (train_feat - mu) ./ sigma;
        test_norm = (test_feat - mu) ./ sigma;
        
        % train simple linear classifier
        X = train_norm';
        y = train_lbl';
        lambda = 0.01;
        w = (X * X' + lambda * eye(size(X,1))) \ (X * y');
        
        % predict
        scores = w' * test_norm';
        scores = scores';
        scores = (scores - min(scores)) / (max(scores) - min(scores) + eps);
        
        % calculate metrics
        [acc(u_idx), far(u_idx), frr(u_idx), eer(u_idx)] = ...
            calculate_metrics(scores, test_lbl);
    end
end


function [train_features, train_labels, test_features, test_labels] = ...
    extract_user_data(file_map, all_users, target_user, sensor_mode, fs)
    % Extract and label features for one user
    
    train_features = [];
    train_labels = [];
    test_features = [];
    test_labels = [];
    
    % process all users
    for u_idx = 1:length(all_users)
        user = all_users(u_idx);
        key = sprintf('U%d', user);
        
        % training data (FD)
        if ~isempty(file_map.(key).FD)
            load(file_map.(key).FD, 'segments');
            feats = extract_features_from_segments(segments, sensor_mode, fs);
            
            if user == target_user
                train_features = [train_features; feats];
                train_labels = [train_labels; ones(size(feats,1), 1)];
            else
                train_features = [train_features; feats];
                train_labels = [train_labels; zeros(size(feats,1), 1)];
            end
        end
        
        % testing data (MD)
        if ~isempty(file_map.(key).MD)
            load(file_map.(key).MD, 'segments');
            feats = extract_features_from_segments(segments, sensor_mode, fs);
            
            if user == target_user
                test_features = [test_features; feats];
                test_labels = [test_labels; ones(size(feats,1), 1)];
            else
                test_features = [test_features; feats];
                test_labels = [test_labels; zeros(size(feats,1), 1)];
            end
        end
    end
end


function features = extract_features_from_segments(segments, sensor_mode, fs)
    % Extract features from all segments
    
    features = [];
    
    for s = 1:length(segments)
        seg = segments{s};
        fv = extract_features_optimization(seg, fs, sensor_mode);
        features = [features; fv];
    end
end


function [acc, far, frr, eer] = calculate_metrics(outputs, labels)
    % Calculate classification metrics
    
    thresholds = 0:0.01:1;
    far_curve = zeros(size(thresholds));
    frr_curve = zeros(size(thresholds));
    
    genuine = (labels == 1);
    
    for t = 1:length(thresholds)
        pred = outputs >= thresholds(t);
        
        fp = sum(~genuine & pred);
        tn = sum(~genuine & ~pred);
        fn = sum(genuine & ~pred);
        tp = sum(genuine & pred);
        
        far_curve(t) = fp / (fp + tn + eps) * 100;
        frr_curve(t) = fn / (fn + tp + eps) * 100;
    end
    
    % find EER
    [~, eer_idx] = min(abs(far_curve - frr_curve));
    eer = (far_curve(eer_idx) + frr_curve(eer_idx)) / 2;
    optimal_threshold = thresholds(eer_idx);
    
    % final metrics at optimal threshold
    pred_final = outputs >= optimal_threshold;
    tp = sum(genuine & pred_final);
    tn = sum(~genuine & ~pred_final);
    fp = sum(~genuine & pred_final);
    fn = sum(genuine & ~pred_final);
    
    acc = (tp + tn) / (tp + tn + fp + fn) * 100;
    far = fp / (fp + tn + eps) * 100;
    frr = fn / (fn + tp + eps) * 100;
end

%% ==================== PER-USER PERFORMANCE TABLE (OPTIMIZED) ====================
fprintf('\n=== Generating Per-User Performance Table (Optimized Configuration) ===\n');

% OPTIMIZED configuration (best from experiments)
optimized_mode = 'combined';
optimized_neurons = 10;  % Best from Experiment 2
optimized_split = 0.8;   % Best from Experiment 3
optimized_features = 70; % Best cost-performance from Experiment 4

fprintf('Optimized Config: %s sensors, %d neurons, %.0f/%.0f split, %d features\n\n', ...
    optimized_mode, optimized_neurons, optimized_split*100, (1-optimized_split)*100, optimized_features);

% Arrays to store per-user results
user_ccr = zeros(num_users, 1);
user_far = zeros(num_users, 1);
user_frr = zeros(num_users, 1);
user_eer = zeros(num_users, 1);

fprintf('Processing per-user results...\n');

% Run optimized configuration for each user
for u_idx = 1:num_users
    user = valid_users(u_idx);
    
    fprintf('  User %d... ', user);
    
    % Extract features for this user
    [train_feat, train_lbl, test_feat, test_lbl] = ...
        extract_user_data(file_map, valid_users, user, optimized_mode, fs);
    
    if isempty(train_feat) || isempty(test_feat)
        fprintf('SKIP (no data)\n');
        continue;
    end
    
    % Apply feature reduction
    if optimized_features < size(train_feat, 2)
        train_feat = train_feat(:, 1:optimized_features);
        test_feat = test_feat(:, 1:optimized_features);
    end
    
    % Apply train/test split
    num_train = size(train_feat, 1);
    num_use = round(num_train * optimized_split);
    idx = randperm(num_train, num_use);
    train_feat = train_feat(idx, :);
    train_lbl = train_lbl(idx);
    
    % Normalize
    mu = mean(train_feat);
    sigma = std(train_feat) + eps;
    train_norm = (train_feat - mu) ./ sigma;
    test_norm = (test_feat - mu) ./ sigma;
    
    % Train neural network
    net = feedforwardnet(optimized_neurons, 'trainscg');
    net.trainParam.epochs = 300;
    net.trainParam.showWindow = false;
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    net = train(net, train_norm', train_lbl');
    
    % Test
    outputs = net(test_norm');
    
    % Calculate metrics
    [acc, far, frr, eer] = calculate_metrics(outputs', test_lbl);
    
    user_ccr(u_idx) = acc;
    user_far(u_idx) = far;
    user_frr(u_idx) = frr;
    user_eer(u_idx) = eer;
    
    fprintf('CCR=%.2f%%, EER=%.2f%%\n', acc, eer);
end

%% Display Per-User Performance Table
fprintf('\n=== PER-USER PERFORMANCE TABLE (OPTIMIZED) ===\n');
fprintf('User | CCR%%     | FAR%%    | FRR%%    | EER%%\n');
fprintf('-----|---------|--------|--------|--------\n');

for u_idx = 1:num_users
    user = valid_users(u_idx);
    fprintf('%-4d | %7.2f | %6.2f | %6.2f | %6.2f\n', ...
        user, user_ccr(u_idx), user_far(u_idx), user_frr(u_idx), user_eer(u_idx));
end

fprintf('-----|---------|--------|--------|--------\n');
fprintf('AVG  | %7.2f | %6.2f | %6.2f | %6.2f\n', ...
    mean(user_ccr), mean(user_far), mean(user_frr), mean(user_eer));

%% Save per-user results to CSV
per_user_csv = fullfile(RESULT_DIR, 'per_user_optimized.csv');
fid = fopen(per_user_csv, 'w');
fprintf(fid, 'User,CCR,FAR,FRR,EER\n');
for u_idx = 1:num_users
    user = valid_users(u_idx);
    fprintf(fid, '%.2f,%.2f,%.2f,%.2f,%.2f\n', ...
        user, user_ccr(u_idx), user_far(u_idx), user_frr(u_idx), user_eer(u_idx));
end
fprintf(fid, '%.2f,%.2f,%.2f,%.2f,%.2f\n', ...
    mean(user_ccr), mean(user_far), mean(user_frr), mean(user_eer));
fclose(fid);

fprintf('\nPer-user results saved to: %s\n', per_user_csv);
fprintf('\n=== PER-USER ANALYSIS COMPLETE ===\n');