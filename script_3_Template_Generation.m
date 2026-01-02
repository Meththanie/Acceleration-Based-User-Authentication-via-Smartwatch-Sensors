% script3_Template_Generation.m
% -----------------------------------------
% Creates templates for EACH USER:
%  - Genuine Train (FD)
%  - Impostor Train (others FD)
%  - Genuine Test (MD)
%  - Impostor Test (others MD)

clc; clear; close all;

FEATURE_DIR   = fullfile(pwd, 'Features');
TEMPLATE_DIR  = fullfile(pwd, 'Templates');

if ~exist(TEMPLATE_DIR,'dir')
    mkdir(TEMPLATE_DIR);
end

fprintf('\n=== TEMPLATE GENERATION STARTED ===\n');

% Load all extracted feature files
files = dir(fullfile(FEATURE_DIR, '*_features.mat'));
if isempty(files)
    error('No feature files found in %s', FEATURE_DIR);
end

% Identify FD and MD files
FD_files = {};
MD_files = {};
FD_users = [];
MD_users = [];

for i = 1:length(files)
    fname = files(i).name;

    % Detect user number (U1, U2, ...)
    token = regexp(fname, 'U(\d+)', 'tokens');
    if isempty(token), continue; end
    userID = str2double(token{1}{1});

    % Separate FD/MD
    if contains(fname, '_FD')
        FD_files{end+1} = fname;
        FD_users(end+1) = userID;

    elseif contains(fname, '_MD')
        MD_files{end+1} = fname;
        MD_users(end+1) = userID;
    end
end

users = unique([FD_users MD_users]);
num_users = length(users);

fprintf('Found %d users\n', num_users);
fprintf('FD files: %d | MD files: %d\n\n', length(FD_files), length(MD_files));

% ---------------------------------------------------------
% CREATE TEMPLATE FOR EACH USER
% ---------------------------------------------------------
for u = 1:num_users
    UID = users(u);
    fprintf('[%d/%d] Creating template for User %d\n', u, num_users, UID);

    train_features = [];
    train_labels   = [];
    test_features  = [];
    test_labels    = [];

    % --------------------------
    % TRAINING (FD)
    % --------------------------
    for i = 1:length(FD_files)
        fname = FD_files{i};
        user_i = FD_users(i);

        load(fullfile(FEATURE_DIR, fname), "all_features");

        if user_i == UID
            % Genuine samples for this user
            train_features = [train_features; all_features];
            train_labels   = [train_labels; ones(size(all_features,1),1)];
        else
            % Impostor samples
            train_features = [train_features; all_features];
            train_labels   = [train_labels; zeros(size(all_features,1),1)];
        end
    end

    % --------------------------
    % TESTING (MD)
    % --------------------------
    for i = 1:length(MD_files)
        fname = MD_files{i};
        user_i = MD_users(i);

        load(fullfile(FEATURE_DIR, fname), "all_features");

        if user_i == UID
            test_features = [test_features; all_features];
            test_labels   = [test_labels; ones(size(all_features,1),1)];
        else
            test_features = [test_features; all_features];
            test_labels   = [test_labels; zeros(size(all_features,1),1)];
        end
    end

    fprintf('  Training: %d genuine, %d impostor\n', ...
        sum(train_labels==1), sum(train_labels==0));

    fprintf('  Testing:  %d genuine, %d impostor\n', ...
        sum(test_labels==1), sum(test_labels==0));

    % --------------------------
    % SAVE TEMPLATE
    % --------------------------
    save(fullfile(TEMPLATE_DIR, sprintf('User%d_template.mat', UID)), ...
         'train_features', 'train_labels', 'test_features', 'test_labels');

    fprintf('  Saved User%d_template.mat\n\n', UID);
end

fprintf('=== TEMPLATE GENERATION COMPLETE ===\n');
fprintf('Templates saved in: %s\n\n', TEMPLATE_DIR);
