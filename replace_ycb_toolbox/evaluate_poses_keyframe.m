function evaluate_poses_keyframe

opt = globals();

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
object_names = C{1};
fclose(fid);

% load model points
num_objects = numel(object_names);
models = cell(num_objects, 1);
for i = 1:num_objects
    filename = fullfile(opt.root, 'models', object_names{i}, 'points.xyz');
    disp(filename);
    models{i} = load(filename);
end

% load the keyframe indexes
fid = fopen('keyframe.txt', 'r');
C = textscan(fid, '%s');
keyframes = C{1};
fclose(fid);

% save results
distances_sys = zeros(100000, 5);
distances_non = zeros(100000, 5);
errors_rotation = zeros(100000, 5); 
errors_translation = zeros(100000, 5);
results_seq_id = zeros(100000, 1);
results_frame_id = zeros(100000, 1);
results_object_id = zeros(100000, 1);
results_cls_id = zeros(100000, 1);

% for each image
count = 0;
for i = 1:numel(keyframes)
    
    % parse keyframe name
    name = keyframes{i};
    pos = strfind(name, '/');
    seq_id = str2double(name(1:pos-1));
    frame_id = str2double(name(pos+1:end));
            
    % load PoseCNN result
    filename = sprintf('results_PoseCNN_RSS2018/%06d.mat', i - 1);
    result = load(filename);
    filename = sprintf('Densefusion_iterative_result/%04d.mat', i - 1);
    result_my = load(filename);
    filename = sprintf('Densefusion_wo_refine_result/%04d.mat', i - 1);
    result_mygt = load(filename);

    % load 3D coordinate regression result
    filename = sprintf('results_3DCoordinate/%04d.mat', i - 1);
    result_3DCoordinate = load(filename);

    % load gt poses
    filename = fullfile(opt.root, 'data', sprintf('%04d/%06d-meta.mat', seq_id, frame_id));
    disp(filename);
    gt = load(filename);

    % for each gt poses
    for j = 1:numel(gt.cls_indexes)
        count = count + 1;
        cls_index = gt.cls_indexes(j);
        RT_gt = gt.poses(:, :, j);

        results_seq_id(count) = seq_id;
        results_frame_id(count) = frame_id;
        results_object_id(count) = j;
        results_cls_id(count) = cls_index;

        % network result
        roi_index = find(result.rois(:, 2) == cls_index);
        if isempty(roi_index) == 0
            RT = zeros(3, 4);

            % pose from network
            RT(1:3, 1:3) = quat2rotm(result_my.poses(roi_index, 1:4));
            RT(:, 4) = result_my.poses(roi_index, 5:7);
            distances_sys(count, 1) = adi(RT, RT_gt, models{cls_index}');
            distances_non(count, 1) = add(RT, RT_gt, models{cls_index}');
            errors_rotation(count, 1) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            errors_translation(count, 1) = te(RT(:, 4), RT_gt(:, 4));

            % pose after ICP refinement
            RT(1:3, 1:3) = quat2rotm(result.poses_icp(roi_index, 1:4));
            RT(:, 4) = result.poses_icp(roi_index, 5:7);
            distances_sys(count, 2) = adi(RT, RT_gt, models{cls_index}');
            distances_non(count, 2) = add(RT, RT_gt, models{cls_index}');
            errors_rotation(count, 2) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            errors_translation(count, 2) = te(RT(:, 4), RT_gt(:, 4));

            % pose from multiview
            RT(1:3, 1:3) = quat2rotm(result_mygt.poses(roi_index, 1:4));
            RT(:, 4) = result_mygt.poses(roi_index, 5:7);
            distances_sys(count, 3) = adi(RT, RT_gt, models{cls_index}');
            distances_non(count, 3) = add(RT, RT_gt, models{cls_index}');
            errors_rotation(count, 3) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            errors_translation(count, 3) = te(RT(:, 4), RT_gt(:, 4));
% 
            
%             % pose from multiview + ICP
%             RT(1:3, 1:3) = quat2rotm(result.poses_multiview_icp(roi_index, 1:4));
%             RT(:, 4) = result.poses_multiview_icp(roi_index, 5:7);
%             distances_sys(count, 4) = adi(RT, RT_gt, models{cls_index}');
%             distances_non(count, 4) = add(RT, RT_gt, models{cls_index}');
%             errors_rotation(count, 4) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
%             errors_translation(count, 4) = te(RT(:, 4), RT_gt(:, 4));                    
        else
            distances_sys(count, 1:4) = inf;
            distances_non(count, 1:4) = inf;
            errors_rotation(count, 1:4) = inf;
            errors_translation(count, 1:4) = inf;
        end


        % 3D Coordinate regression result
        roi_index = find(result_3DCoordinate.rois(:, 2) == cls_index);
        if isempty(roi_index) == 0
            RT = zeros(3, 4);
            RT(1:3, 1:3) = quat2rotm(result_3DCoordinate.poses(roi_index, 1:4));
            RT(:, 4) = result_3DCoordinate.poses(roi_index, 5:7);
            distances_sys(count, 5) = adi(RT, RT_gt, models{cls_index}');
            distances_non(count, 5) = add(RT, RT_gt, models{cls_index}');
            errors_rotation(count, 5) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            errors_translation(count, 5) = te(RT(:, 4), RT_gt(:, 4));                    
        else
            distances_sys(count, 5) = inf;
            distances_non(count, 5) = inf;
            errors_rotation(count, 5) = inf;
            errors_translation(count, 5) = inf;
        end
    end
end
distances_sys = distances_sys(1:count, :);
distances_non = distances_non(1:count, :);
errors_rotation = errors_rotation(1:count, :);
errors_translation = errors_translation(1:count, :);
results_seq_id = results_seq_id(1:count);
results_frame_id = results_frame_id(1:count);
results_object_id = results_object_id(1:count, :);
results_cls_id = results_cls_id(1:count, :);
save('results_keyframe.mat', 'distances_sys', 'distances_non', 'errors_rotation', 'errors_translation',...
    'results_seq_id', 'results_frame_id', 'results_object_id', 'results_cls_id');

function pts_new = transform_pts_Rt(pts, RT)
%     """
%     Applies a rigid transformation to 3D points.
% 
%     :param pts: nx3 ndarray with 3D points.
%     :param R: 3x3 rotation matrix.
%     :param t: 3x1 translation vector.
%     :return: nx3 ndarray with transformed 3D points.
%     """
n = size(pts, 2);
pts_new = RT * [pts; ones(1, n)];

function error = add(RT_est, RT_gt, pts)
%     """
%     Average Distance of Model Points for objects with no indistinguishable views
%     - by Hinterstoisser et al. (ACCV 2012).
% 
%     :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param model: Object model given by a dictionary where item 'pts'
%     is nx3 ndarray with 3D model points.
%     :return: Error of pose_est w.r.t. pose_gt.
%     """
pts_est = transform_pts_Rt(pts, RT_est);
pts_gt = transform_pts_Rt(pts, RT_gt);
diff = pts_est - pts_gt;
error = mean(sqrt(sum(diff.^2, 1)));

function error = adi(RT_est, RT_gt, pts)
%     """
%     Average Distance of Model Points for objects with indistinguishable views
%     - by Hinterstoisser et al. (ACCV 2012).
% 
%     :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param model: Object model given by a dictionary where item 'pts'
%     is nx3 ndarray with 3D model points.
%     :return: Error of pose_est w.r.t. pose_gt.
%     """
pts_est = transform_pts_Rt(pts, RT_est);
pts_gt = transform_pts_Rt(pts, RT_gt);

% Calculate distances to the nearest neighbors from pts_gt to pts_est
MdlKDT = KDTreeSearcher(pts_est');
[~, D] = knnsearch(MdlKDT, pts_gt');
error = mean(D);

function error = re(R_est, R_gt)
%     """
%     Rotational Error.
% 
%     :param R_est: Rotational element of the estimated pose (3x1 vector).
%     :param R_gt: Rotational element of the ground truth pose (3x1 vector).
%     :return: Error of t_est w.r.t. t_gt.
%     """

error_cos = 0.5 * (trace(R_est * inv(R_gt)) - 1.0);
error_cos = min(1.0, max(-1.0, error_cos));
error = acos(error_cos);
error = 180.0 * error / pi;

function error = te(t_est, t_gt)
% """
% Translational Error.
% 
% :param t_est: Translation element of the estimated pose (3x1 vector).
% :param t_gt: Translation element of the ground truth pose (3x1 vector).
% :return: Error of t_est w.r.t. t_gt.
% """
error = norm(t_gt - t_est);