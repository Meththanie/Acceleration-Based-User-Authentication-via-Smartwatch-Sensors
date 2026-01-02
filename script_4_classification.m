%  script_4_classification.m   
%  One Binary Classifier Per User 
%  Outputs: Accuracy, FAR, FRR, EER + Visualizations

clc; clear; close all;
rng(42);

TEMPLATE_DIR = fullfile(pwd, 'Templates');
RESULT_DIR   = fullfile(pwd, 'Results');
if ~exist(RESULT_DIR, 'dir'), mkdir(RESULT_DIR); end

fprintf("\n=== SCRIPT 4: CLASSIFICATION STARTED ===\n");

%% ------------------ Neural Network Settings -------------------
hidden_neurons  = 30;         
train_algorithm = 'trainscg'; % Best default SCG
max_epochs      = 300;

%% ------------------ Load Templates ----------------------------
template_files = dir(fullfile(TEMPLATE_DIR, 'User*_template.mat'));
num_users = length(template_files);

if num_users == 0
    error("No templates found in %s", TEMPLATE_DIR);
end

fprintf("Found %d user templates.\n", num_users);

%% ------------------ Init result arrays ------------------------
ACC  = zeros(num_users,1);
PREC = zeros(num_users,1);
REC  = zeros(num_users,1);
FAR  = zeros(num_users,1);
FRR  = zeros(num_users,1);
EER  = zeros(num_users,1);
ThR  = zeros(num_users,1);

AllResults = struct();

%% ============================================================
%        TRAIN SEPARATE BINARY CLASSIFIER FOR EACH USER
% =============================================================
for u = 1:num_users

    fprintf("\n[%d/%d] User %d\n", u, num_users, u);

    load(fullfile(TEMPLATE_DIR, template_files(u).name), ...
        "train_features", "train_labels", "test_features", "test_labels");

    fprintf("Training samples: %d\n", length(train_labels));

    %% ----------- Normalization (Train Stats Only) -------------
    mu = mean(train_features);
    sd = std(train_features);

    trainN = (train_features - mu) ./ (sd + eps);
    testN  = (test_features  - mu) ./ (sd + eps);

    Xtr = trainN';  Ytr = train_labels';
    Xte = testN';   Yte = test_labels';

    %% ----------- Create Neural Network -------------------------
    net = feedforwardnet(hidden_neurons, train_algorithm);
    net.trainParam.epochs = max_epochs;
    net.trainParam.showWindow = false;
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio   = 0.15;
    net.divideParam.testRatio  = 0.15;

    [net,~] = train(net, Xtr, Ytr);

    %% ----------- Testing --------------------------------------
    out = net(Xte);

    thresholds = 0:0.01:1;
    FAR_curve = zeros(size(thresholds));
    FRR_curve = zeros(size(thresholds));

    genuine = (Yte == 1);

    for t = 1:length(thresholds)
        pred = out >= thresholds(t);

        FP = sum(~genuine & pred);
        TN = sum(~genuine & ~pred);
        FN = sum(genuine & ~pred);
        TP = sum(genuine & pred);

        FAR_curve(t) = FP / (FP + TN + eps) * 100;
        FRR_curve(t) = FN / (FN + TP + eps) * 100;
    end

    % ---- Compute Equal Error Rate ----
    [~, idx] = min(abs(FAR_curve - FRR_curve));
    EER(u) = (FAR_curve(idx) + FRR_curve(idx)) / 2;
    ThR(u) = thresholds(idx);

    % Metrics at optimal threshold
    pred_final = out >= ThR(u);

    TP = sum(genuine & pred_final);
    TN = sum(~genuine & ~pred_final);
    FP = sum(~genuine & pred_final);
    FN = sum(genuine & ~pred_final);

    ACC(u)  = (TP + TN) / (TP + TN + FP + FN) * 100;
    PREC(u) = TP / (TP + FP + eps) * 100;
    REC(u)  = TP / (TP + FN + eps) * 100;
    FAR(u)  = FP / (FP + TN + eps) * 100;
    FRR(u)  = FN / (FN + TP + eps) * 100;

    % Save result struct
    AllResults(u).User       = u;
    AllResults(u).Accuracy   = ACC(u);
    AllResults(u).Precision  = PREC(u);
    AllResults(u).Recall     = REC(u);
    AllResults(u).FAR        = FAR(u);
    AllResults(u).FRR        = FRR(u);
    AllResults(u).EER        = EER(u);
    AllResults(u).Threshold  = ThR(u);
    AllResults(u).FAR_curve  = FAR_curve;
    AllResults(u).FRR_curve  = FRR_curve;
    AllResults(u).Thresholds = thresholds;

end

%% ============================================================
%                    SAVE RESULTS SUMMARY
% =============================================================
save(fullfile(RESULT_DIR, 'ClassificationResults.mat'), ...
     'ACC','PREC','REC','FAR','FRR','EER','ThR','AllResults');

fprintf("\n=== SUMMARY (ALL USERS) ===\n");
fprintf("Average Accuracy: %.2f%%\n", mean(ACC));
fprintf("Average EER:      %.2f%%\n", mean(EER));
fprintf("Lowest EER:       %.2f%% (User %d)\n", min(EER), find(EER==min(EER)));
fprintf("Highest EER:      %.2f%% (User %d)\n", max(EER), find(EER==max(EER)));


%% ============================================================
%                     VISUALIZATION SECTION
% =============================================================

fprintf("\n=== GENERATING VISUALIZATIONS ===\n");

%% ------------------ (1) Bar Charts ---------------------------
figure('Position',[100 100 1200 700]);

subplot(2,2,1); bar(ACC);  title('Accuracy per User'); ylabel('%'); ylim([0 100]); grid on;
subplot(2,2,2); bar(FAR);  title('FAR per User'); ylabel('%'); ylim([0 10]); grid on;
subplot(2,2,3); bar(FRR);  title('FRR per User'); ylabel('%'); ylim([0 10]); grid on;
subplot(2,2,4); bar(EER);  title('EER per User'); ylabel('%'); ylim([0 10]); grid on;

saveas(gcf, fullfile(RESULT_DIR, 'PerUser_BarCharts.png'));

%% ------------------ (2) FAR/FRR Curves ------------------------
figure('Position',[100 100 1300 700]);
for u = 1:num_users
    subplot(2,5,u);
    plot(AllResults(u).Thresholds, AllResults(u).FAR_curve, 'r', 'LineWidth',1.5); hold on;
    plot(AllResults(u).Thresholds, AllResults(u).FRR_curve, 'b', 'LineWidth',1.5);

    [~,idx] = min(abs(AllResults(u).FAR_curve - AllResults(u).FRR_curve));
    plot(AllResults(u).Thresholds(idx), AllResults(u).FAR_curve(idx), 'ko','MarkerSize',8,'LineWidth',1.5);

    title(sprintf('User %d (EER=%.2f%%)', u, AllResults(u).EER));
    xlabel('Threshold'); ylabel('Error (%)'); grid on;
    legend('FAR','FRR','EER Point');
end
saveas(gcf, fullfile(RESULT_DIR, 'FAR_FRR_AllUsers.png'));

%% ------------------ (3) ROC Curves ---------------------------
figure('Position',[100 100 1300 700]);
for u = 1:num_users
    subplot(2,5,u);
    TPR = 100 - AllResults(u).FRR_curve;
    FPR = AllResults(u).FAR_curve;

    plot(FPR, TPR, 'b', 'LineWidth',1.5); hold on;
    plot([0 100],[0 100],'r--');

    title(sprintf('User %d ROC', u));
    xlabel('FPR (%)'); ylabel('TPR (%)');
    axis([0 100 0 100]); axis square; grid on;
end
saveas(gcf, fullfile(RESULT_DIR, 'ROC_AllUsers.png'));

%% ------------------ (4) Average FAR/FRR ----------------------
avg_far = mean(cell2mat({AllResults.FAR_curve}'),1);
avg_frr = mean(cell2mat({AllResults.FRR_curve}'),1);

figure('Position',[100 100 800 600]);
plot(thresholds, avg_far,'r','LineWidth',2); hold on;
plot(thresholds, avg_frr,'b','LineWidth',2);

[~,best] = min(abs(avg_far - avg_frr));
avg_eer = (avg_far(best) + avg_frr(best))/2;

plot(thresholds(best), avg_eer,'ko','MarkerSize',10,'LineWidth',2);

title(sprintf('Average FAR/FRR (EER = %.2f%%)', avg_eer));
xlabel('Threshold'); ylabel('Error (%)'); grid on;
legend('Average FAR','Average FRR','EER Point');
saveas(gcf, fullfile(RESULT_DIR, 'Average_FAR_FRR.png'));

%% ------------------ (5) Average ROC Curve --------------------
avg_TPR = 100 - avg_frr;
avg_FPR = avg_far;

AUC = trapz(avg_FPR, avg_TPR) / 10000 * 100;

figure('Position',[100 100 800 600]);
plot(avg_FPR, avg_TPR, 'b','LineWidth',2.5); hold on;
plot([0 100],[0 100],'r--');

title(sprintf('Average ROC Curve (AUC = %.2f%%)', AUC));
xlabel('False Positive Rate (%)'); ylabel('True Positive Rate (%)');
axis([0 100 0 100]); axis square; grid on;

saveas(gcf, fullfile(RESULT_DIR, 'Average_ROC.png'));

fprintf("\n=== VISUALIZATION COMPLETE ===\n");
fprintf("All plots saved in: %s\n", RESULT_DIR);

%% ---------------- CREATE USER METRIC TABLE ---------------------
fprintf("\n=== PER-USER PERFORMANCE TABLE ===\n");

fprintf('%-8s | %-8s | %-8s | %-8s | %-8s\n', ...
        'User', 'CCR%', 'FAR%', 'FRR%', 'EER%');
fprintf('%s\n', repmat('-',1,50));

for u = 1:num_users
    fprintf('%-8d | %-8.2f | %-8.2f | %-8.2f | %-8.2f\n', ...
        u, ACC(u), FAR(u), FRR(u), EER(u));
end

% Save the table to CSV
T = table((1:num_users)', ACC, FAR, FRR, EER, ...
        'VariableNames', {'User','CCR','FAR','FRR','EER'});

writetable(T, fullfile(RESULT_DIR, 'PerUser_Metrics.csv'));

fprintf('\nSaved per-user table as PerUser_Metrics.csv\n');

fprintf("\n=== SCRIPT 4 COMPLETE ===\n");
