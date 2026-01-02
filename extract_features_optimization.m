function fv = extract_features_optimization(seg, fs, sensor_mode)
% extract features from segment - optimization version
% seg = data segment, fs = sampling rate, sensor_mode = which sensors to use

if nargin < 3
    sensor_mode = 'combined';
end

fv = [];
[N, C] = size(seg);

% figure out which columns to use
if strcmp(sensor_mode, 'acc_only')
    use_cols = 1:3;
    has_acc = true;
    has_gyro = false;
elseif strcmp(sensor_mode, 'gyro_only')
    use_cols = 4:6;
    has_acc = false;
    has_gyro = true;
else
    use_cols = 1:6;
    has_acc = true;
    has_gyro = true;
end

% time domain features for each column
for c = use_cols
    x = seg(:,c);
    
    m = mean(x);
    s = std(x);
    v = var(x);
    minval = min(x);
    maxval = max(x);
    rng = maxval - minval;
    
    p25 = prctile(x,25);
    p50 = prctile(x,50);
    p75 = prctile(x,75);
    iqr = p75 - p25;
    
    if s > 0
        skew = mean(((x-m)/s).^3);
        kurt = mean(((x-m)/s).^4);
    else
        skew = 0;
        kurt = 0;
    end
    
    rmsval = rms(x);
    energy = sum(x.^2);
    zcr = sum(abs(diff(sign(x))))/(N-1);
    mad = mean(abs(x-m));
    
    fv = [fv, m, s, v, minval, maxval, rng, p25, p50, p75, iqr, skew, kurt, rmsval, energy, zcr, mad];
end

% magnitude and correlation features
if has_acc
    acc = seg(:,1:3);
    accmag = sqrt(sum(acc.^2,2));
    
    sma_a = sum(abs(accmag))/N;
    smv_a = mean(accmag);
    
    cxy = corrcoef_manual(acc(:,1), acc(:,2));
    cxz = corrcoef_manual(acc(:,1), acc(:,3));
    cyz = corrcoef_manual(acc(:,2), acc(:,3));
    
    % autocorr
    lag = min(20, N-1);
    ac = zeros(1,lag);
    for k = 1:lag
        ac(k) = corrcoef_manual(accmag(1:end-k), accmag(1+k:end));
    end
    
    acmax = max(ac);
    acmean = mean(ac);
    
    fv = [fv, sma_a, smv_a, cxy, cxz, cyz, acmax, acmean];
end

if has_gyro
    gyro = seg(:,4:6);
    gyromag = sqrt(sum(gyro.^2,2));
    
    sma_g = sum(abs(gyromag))/N;
    smv_g = mean(gyromag);
    
    fv = [fv, sma_g, smv_g];
end

fv = double(fv);
end


function r = corrcoef_manual(x,y)
x = x(:);
y = y(:);

valid = ~isnan(x) & ~isnan(y);
x = x(valid);
y = y(valid);

if length(x) < 2
    r = 0;
    return;
end

mx = mean(x);
my = mean(y);

dx = x - mx;
dy = y - my;

num = sum(dx.*dy);
den = sqrt(sum(dx.^2)*sum(dy.^2));

if den == 0
    r = 0;
else
    r = num/den;
end
end