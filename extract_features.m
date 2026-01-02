function fv = extract_features(seg, fs)
% FEATURE EXTRACTION FOR SMARTWATCH GAIT DATA
% Returns 105 time-domain features (appropriate for smartwatch sensors)
% Input: seg - segment matrix (150 x 6) with AccXYZ and GyroXYZ
% Output: fv - feature vector (1 x 105)
% NO TOOLBOX DEPENDENCIES - Uses manual correlation calculation

fv = [];
[N, C] = size(seg);

%% =====================================================
% TIME-DOMAIN FEATURES (96 features = 16 per channel x 6)
% ======================================================
for c = 1:C
    x = seg(:,c);

    m  = mean(x);
    sd = std(x);
    v  = var(x);
    mn = min(x);
    mx = max(x);
    rg = mx - mn;

    p25 = prctile(x,25);
    p50 = prctile(x,50);
    p75 = prctile(x,75);
    iqr_val = p75 - p25;

    if sd > 0
        skew_val = mean(((x-m)/sd).^3);
        kurt_val = mean(((x-m)/sd).^4);
    else
        skew_val = 0; 
        kurt_val = 0;
    end

    rms_val = rms(x);
    energy  = sum(x.^2);
    zcr     = sum(abs(diff(sign(x)))) / (N-1);
    mad_val = mean(abs(x-m));

    fv = [fv, m, sd, v, mn, mx, rg, ...
          p25, p50, p75, iqr_val, ...
          skew_val, kurt_val, ...
          rms_val, energy, zcr, mad_val];
end

%% =====================================================
% GLOBAL GAIT FEATURES (9 features)
% ======================================================
acc  = seg(:,1:3);
gyro = seg(:,4:6);

accMag  = sqrt(sum(acc.^2,2));
gyroMag = sqrt(sum(gyro.^2,2));

SMA_acc  = sum(abs(accMag))  / N;
SMA_gyro = sum(abs(gyroMag)) / N;

SMV_acc  = mean(accMag);
SMV_gyro = mean(gyroMag);

% Manual correlation calculation (no toolbox needed)
corr_xy = manual_corr(acc(:,1), acc(:,2));
corr_xz = manual_corr(acc(:,1), acc(:,3));
corr_yz = manual_corr(acc(:,2), acc(:,3));

% Manual autocorrelation
maxLag = min(20, N-1);
ac = zeros(1, maxLag);

for k = 1:maxLag
    ac(k) = manual_corr(accMag(1:end-k), accMag(1+k:end));
end

auto_max  = max(ac);
auto_mean = mean(ac);

fv = [fv, SMA_acc, SMV_acc, SMA_gyro, SMV_gyro, ...
          corr_xy, corr_xz, corr_yz, ...
          auto_max, auto_mean];

fv = double(fv);
end

%% =====================================================
% HELPER FUNCTION: Manual Correlation
% ======================================================
function r = manual_corr(x, y)
    % Manual Pearson correlation coefficient
    % r = cov(x,y) / (std(x) * std(y))
    
    x = x(:);  % Make column vector
    y = y(:);
    
    % Remove NaN values
    valid = ~isnan(x) & ~isnan(y);
    x = x(valid);
    y = y(valid);
    
    if length(x) < 2
        r = 0;
        return;
    end
    
    % Calculate means
    mx = mean(x);
    my = mean(y);
    
    % Calculate deviations
    dx = x - mx;
    dy = y - my;
    
    % Calculate correlation
    numerator = sum(dx .* dy);
    denominator = sqrt(sum(dx.^2) * sum(dy.^2));
    
    if denominator == 0
        r = 0;
    else
        r = numerator / denominator;
    end
end