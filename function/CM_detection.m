function [p1,p2] = CM_detection(im,para)
    %% Init
    h = para.h;
    w = para.w;
    step1 = para.step1;
    step2 = para.step2;
    step3 = para.step3;
    step4 = para.step4;
    t1 = para.t1;
    t2 = para.t2;
    thre = para.thre;
    
    %% Keypoint detection and feature extraction
    E = entropyfilt(uint8(im),true(7));
    locs = vl_sift(single(E));
    [locs,descs] = vl_sift(single(im),'Frames',locs,'Orientations');

    locs = locs';
    descs = double(descs)';
    n_locs = size(locs,1);
    fprintf('Found %d keypoints.\n', n_locs);
    locs(:,1:2) = round(locs(:,1:2));
    locs = [locs,zeros(n_locs,1)];

    for i = 1:n_locs
        temp = descs(i,:);
        abs_temp = temp*temp';
        descs(i,:) = temp/sqrt(abs_temp);
    end

    %% Feature matching
    if n_locs>5000
        key_indx = (locs(:,1)-1)*h+locs(:,2);
        gray_clusters = gray_cluster(im,key_indx,step1,step2,5);
    else
        gray_clusters = {(1:n_locs)'};
    end

    p1 = [];
    p2 = [];
    for i = 1:size(gray_clusters,2)
        idx_gray_cluster = gray_clusters{i};
        locs_gray_cluster = locs(idx_gray_cluster,:);
        descs_gray_cluster = descs(idx_gray_cluster,:);
        n_gray_cluster = size(idx_gray_cluster,1);
        if n_gray_cluster>5000
            key_indx = (locs_gray_cluster(:,1)-1)*h+locs_gray_cluster(:,2);
            entropy_clusters = entropy_cluster(E,key_indx,step3,step4);
        else
            entropy_clusters = {(1:n_gray_cluster)'};
        end
        for j = 1:size(entropy_clusters,2)
            cur_idx = entropy_clusters{j};
            cur_locs = locs_gray_cluster(cur_idx,:);
            cur_descs = descs_gray_cluster(cur_idx,:);
            [match1,match2] = g2NN(cur_locs,cur_descs,cur_locs, cur_descs,thre);
            [pair1,pair2]=match_dis_check(match1,match2,t1);
            p1=[p1;pair1];
            p2=[p2;pair2];
        end
    end

    if size(p1,1) == 0
       return; 
    end

    % Eliminating duplicate matching
    p = round([p1(:,1:2) p2(:,1:2)]);
    [p_temp, indx,~] = unique(p,'rows');
    p1=[p_temp(:,1:2), p1(indx,3:5)];
    p2=[p_temp(:,3:4), p2(indx,3:5)];
    num = size(p1,1);
    p1 = [p1(:,1:2)'; ones(1,num) ; p1(:,3:5)'];
    p2 = [p2(:,1:2)'; ones(1,num) ; p2(:,3:5)'];

    diff = abs(p1(6,:)-p2(6,:));
    idx = find(diff<0.5);
    p1 = p1(:,idx);
    p2 = p2(:,idx);
    n_match = size(p1,2);

    % Removal of isolated matches
    mask = double(circle_mask(t2));
    location_match = zeros(h,w);
    for i = 1:n_match
        y1 = p1(1,i);
        x1 = p1(2,i);
        y2 = p2(1,i);
        x2 = p2(2,i);
        location_match(x1,y1) = location_match(x1,y1)+1;
        location_match(x2,y2) = location_match(x2,y2)+1;
    end
    density_match = imfilter(location_match,mask);
    density = zeros(2,n_match);
    for i = 1:n_match
        y1 = p1(1,i);
        x1 = p1(2,i);
        y2 = p2(1,i);
        x2 = p2(2,i);
        density(1,i) = density_match(x1,y1);
        density(2,i) = density_match(x2,y2);
    end
    density = max(density);
    idx = find(density>3);
    p1 = p1(:,idx);
    p2 = p2(:,idx);

    n_final_match = size(p1,2);
    fprintf('Found %d matches.\n', n_final_match);
end

