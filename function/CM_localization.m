function map = CM_localization(im,para)
    %% Init
    h = para.h;
    w = para.w;
    s = para.s;
    t2 = para.t2;
    p1 = para.p1;
    p2 = para.p2;
    SATS_ratio = para.SATS_ratio;
    min_ransac_inliers = para.min_ransac_inliers;
    min_matches = para.min_matches;
    min_num_SATS = para.min_num_SATS;
    radius = para.radius;
    r_operation = para.r_operation;
    min_size = para.min_size;
    
    %% Post-processing
    t2 = max(min(h,w)/10,t2);
    map1 = zeros(h,w);
    map2 = zeros(h,w);
    n_post = size(p1,2);
    if n_post < 6
        map = imresize(map1,1/s);
        return;
    end
    max_iteration = max([round(n_post/30),15]);
    max_iteration = min([max_iteration,40]);
    visited = zeros(n_post,1);
    for iteration = 1:max_iteration
        unvisited = find(visited == 0);
        n_unvisited = length(unvisited);
        if n_unvisited<4
            break;
        end
        unvisited_p1 = p1(:,unvisited);
        unvisited_p2 = p2(:,unvisited);

        % Decision sampled set
        sample_rand = randi(n_unvisited);
        location = [unvisited_p1(1:2,:),unvisited_p2(1:2,:)];
        diff_p1_x = location(1,:)-unvisited_p1(1,sample_rand);
        diff_p1_y = location(2,:)-unvisited_p1(2,sample_rand);
        diff_p2_x = location(1,:)-unvisited_p2(1,sample_rand);
        diff_p2_y = location(2,:)-unvisited_p2(2,sample_rand);
        diff_p1 = sqrt(diff_p1_x.^2+diff_p1_y.^2);
        diff_p2 = sqrt(diff_p2_x.^2+diff_p2_y.^2);
        good_diff = diff_p1<t2 |diff_p2<t2;
        sample_idx = find(good_diff(1:n_unvisited)|good_diff(n_unvisited+1:end));
        n_sample = length(sample_idx);
        % Remove bad matches
        if n_sample<4
            visited(unvisited(sample_idx)) = 1;
            fprintf('Found few matches in a local region, matches were removed!\n');
            continue;
        end

        cur_p1 = unvisited_p1(:,sample_idx);
        cur_p2 = unvisited_p2(:,sample_idx);
        % Do RANSAC
        t = 0.05;
        [H, inliers_cur] = ransacfithomography2(cur_p1(1:3,:), cur_p2(1:3,:), t);
        % Check homography
        if isempty(H) || length(inliers_cur)<min_ransac_inliers
            visited(unvisited(sample_idx)) = 1;
            continue;
        end
        cur_process_p1 = cur_p1(:,inliers_cur);
        cur_process_p2 = cur_p2(:,inliers_cur);

        % Remove good matches (condition1)
        used = unvisited(sample_idx(inliers_cur));
        visited(used) = 1;

        % Check dominant orientation
        [U,~,V] = svd(H(1:2,1:2));
        rotation_matrix = U*V';
        indx_o  = check_orientation(rotation_matrix(1), rotation_matrix(2),...
            cur_process_p1(5,:), cur_process_p2(5,:), 20);
        if length(indx_o)<size(cur_process_p1,2)*0.8
            fprintf('The dominant orientation was inferior!\n');
            continue;
        end


        % Same Affine Transformation Selection (SATS)
        x1 = p1(1:3,:);
        x2 = p2(1:3,:);
        Hx1 = round(H*x1);
        invHx2 = round(H\x2);
        D2 = sum((x1-invHx2).^2)+sum((x2-Hx1).^2);
        SATS_inliers = find(D2<=8);
        cur_process_p1 = p1(:,SATS_inliers);
        cur_process_p2 = p2(:,SATS_inliers);
        % Remove good matches (condition2)
        visited(SATS_inliers) = 1;

        % Check SATS
        if (size(cur_process_p1,2)>SATS_ratio*n_post && size(cur_process_p1,2)...
                >min_matches)

        elseif size(cur_process_p1,2)>min_num_SATS

        else
            fprintf('check random homography!\n');
            continue;
        end

        % Set localization parameter
        R = radius;
        diff = 16;
        % Localization suspect region1
        R1 = round(R*cur_process_p1(4,:));
        R1(R1>2*R) = 2*R;
        position1 = [cur_process_p1(1:2,:);R1]';
        cur_map1 = zeros(h,w);
        cur_suspect_map1 = insertShape(cur_map1, 'FilledCircle',position1);
        cur_suspect_map1 = imbinarize(cur_suspect_map1(:,:,1));
        [x1,y1] = find(cur_suspect_map1 == 1);
        X1 = [y1,x1,ones(length(x1),1)]';
        X2_ = round(H*X1);
        idx = X2_(1,:)<w & X2_(2,:)<h & X2_(1,:)>0 & X2_(2,:)>0;
        X1 = X1(:,idx);
        X2_ = X2_(:,idx);
        for i = 1:size(X1,2)
            x1 = X1(2,i);
            y1 = X1(1,i);
            x2 = X2_(2,i);
            y2 = X2_(1,i);
            if abs(im(x1,y1)-im(x2,y2))<diff
                cur_map1(x1,y1) = 1;
                cur_map1(x2,y2) = 1;
            end
        end
        map1 = bitor(cur_map1,map1);

        % Localization suspicious region2
        R2 = round(R*cur_process_p2(4,:));
        R2(R2>2*R) = 2*R;
        position2 = [cur_process_p2(1:2,:);R2]';
        cur_map2 = zeros(h,w);
        cur_suspect_map2 = insertShape(cur_map2, 'FilledCircle',position2);
        cur_suspect_map2 = imbinarize(cur_suspect_map2(:,:,1));
        [x2,y2] = find(cur_suspect_map2 == 1);
        X2 = [y2,x2,ones(length(x2),1)]';
        X1_ = round(H\X2);
        idx = X1_(1,:)<w & X1_(2,:)<h & X1_(1,:)>0 & X1_(2,:)>0;
        X2 = X2(:,idx);
        X1_ = X1_(:,idx);
        for i = 1:size(X2,2)
            x2 = X2(2,i);
            y2 = X2(1,i);
            x1 = X1_(2,i);
            y1 = X1_(1,i);
            if abs(im(x1,y1)-im(x2,y2))<diff
                cur_map2(x1,y1) = 1;
                cur_map2(x2,y2) = 1;
            end
        end
        map2 = bitor(cur_map2,map2);
    end

    map = bitor(map1,map2);
    map = bwareaopen(map,min_size);
    map = imclose(map,strel('disk',r_operation));
    map = imopen(map,strel('disk',r_operation));
    map = imresize(map,1/s);
end

