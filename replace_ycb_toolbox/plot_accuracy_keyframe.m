function plot_accuracy_keyframe

color = {'r', 'y', 'g', 'b', 'm'};
leng = {'iterative', 'PoseCNN+ICP', 'per-pixel', '3DCoordinate', ...
    '3D'};
aps = zeros(5, 1);
lengs = cell(5, 1);
close all;

% load results
object = load('results_keyframe.mat');
distances_sys = object.distances_sys;
distances_non = object.distances_non;
rotations = object.errors_rotation;
translations = object.errors_translation;
cls_ids = object.results_cls_id;

index_plot = [2, 3, 1, 5];

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
classes = C{1};
classes{end+1} = 'All 21 objects';
fclose(fid);

hf = figure('units','normalized','outerposition',[0 0 1 1]);
font_size = 12;
max_distance = 0.1;

% for each class
for k = 1:numel(classes)
    index = find(cls_ids == k);
    if isempty(index)
        index = 1:size(distances_sys,1);
    end

    % distance symmetry
    subplot(2, 2, 1);
    for i = index_plot
        D = distances_sys(index, i);
        D(D > max_distance) = inf;
        d = sort(D);
        n = numel(d);
        c = numel(d(d < 0.02));
        accuracy = cumsum(ones(1, n)) / n;
%         fprintf('k = %d i = %d : length %d\n',k,i,length(d));
%         dd = find(d == d(end));
%         ddd = find(d ~= d(end));
%         fprintf('k = %d i = %d : length %d %d %d %d\n',k,i,length(d), length(dd), d(end), ddd(end));

        plot(d, accuracy, color{i}, 'LineWidth', 4);
        aps(i) = VOCap(d, accuracy);
        lengs{i} = sprintf('%s(AUC:%.2f)(<2cm:%.2f)', leng{i}, aps(i)*100, (c/n)*100);
        hold on;
    end
    hold off;
    %h = legend('network', 'refine tranlation only', 'icp', 'stereo translation only', 'stereo full', '3d coordinate');
    %set(h, 'FontSize', 16);
    h = legend(lengs(index_plot), 'Location', 'southeast');
    set(h, 'FontSize', font_size);
    h = xlabel('Average distance threshold in meter (symmetry)');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(classes{k}, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)

    % distance non-symmetry
    subplot(2, 2, 2);
    for i = index_plot
        D = distances_non(index, i);
        D(D > max_distance) = inf;
        d = sort(D);
        n = numel(d);
        c = numel(d(d < 0.02));
        accuracy = cumsum(ones(1, n)) / n;
        plot(d, accuracy, color{i}, 'LineWidth', 4);
        aps(i) = VOCap(d, accuracy);
        lengs{i} = sprintf('%s(AUC:%.2f)(<2cm:%.2f)', leng{i}, aps(i)*100, (c/n)*100);        
        hold on;
    end
    hold off;
    %h = legend('network', 'refine tranlation only', 'icp', 'stereo translation only', 'stereo full', '3d coordinate');
    %set(h, 'FontSize', 16);
    h = legend(lengs(index_plot), 'Location', 'southeast');
    set(h, 'FontSize', font_size);
    h = xlabel('Average distance threshold in meter (non-symmetry)');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(classes{k}, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);    
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)
    
    % rotation
    subplot(2, 2, 3);
    for i = index_plot
        D = rotations(index, i);
        d = sort(D);
        n = numel(d);
        accuracy = cumsum(ones(1, n)) / n;
        plot(d, accuracy, color{i}, 'LineWidth', 4);
        hold on;
    end
    hold off;
    %h = legend('network', 'refine tranlation only', 'icp', 'stereo translation only', 'stereo full', '3d coordinate');
    %set(h, 'FontSize', 16);
    h = legend(leng(index_plot), 'Location', 'southeast');
    set(h, 'FontSize', font_size);
    h = xlabel('Rotation angle threshold');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(classes{k}, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)

    % translation
    subplot(2, 2, 4);
    for i = index_plot
        D = translations(index, i);
        D(D > max_distance) = inf;
        d = sort(D);
        n = numel(d);
        accuracy = cumsum(ones(1, n)) / n;
        plot(d, accuracy, color{i}, 'LineWidth', 4);
        hold on;
    end
    hold off;
    h = legend(leng(index_plot), 'Location', 'southeast');
    set(h, 'FontSize', font_size);
    h = xlabel('Translation threshold in meter');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(classes{k}, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)
    
    filename = sprintf('plots/%s.png', classes{k});
    hgexport(hf, filename, hgexport('factorystyle'), 'Format', 'png');
end

function ap = VOCap(rec, prec)

index = isfinite(rec);
rec = rec(index);
prec = prec(index)';

mrec=[0 ; rec ; 0.1];
% disp(prec)
% disp(end)
% disp(length(prec))
% if length(prec) == 0
%     prec(1) = 1;
% end
% disp(prec(end))

mpre=[0 ; prec ; prec(end)];
for i = 2:numel(mpre)
    mpre(i) = max(mpre(i), mpre(i-1));
end
i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
ap = sum((mrec(i) - mrec(i-1)) .* mpre(i)) * 10;