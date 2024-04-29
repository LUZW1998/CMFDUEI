clear;
clc;
close all
addpath(genpath('function'))
file_path = 'images\29.png';

show_match = 1;
img_rgb=imread(file_path);
s = 2;
img_rgb = imresize(img_rgb,s);
img = rgb2gray(img_rgb);
im = double(img);
[h,w] = size(im);

% Parameters in detection
para.s = s;
para.h = h;
para.w = w;
para.step1 = 40;
para.step2 = 10;
para.step3 = 1;
para.step4 = 0;
para.t1 = max(32*s,min(h,w)/20);
para.t2 = para.t1/2;
para.thre = 0.5;

% Parameters in localization
para.SATS_ratio = 0.2;
para.min_ransac_inliers = 4;
para.min_matches = 10;
para.min_num_SATS = 20;
para.radius = 16;
para.r_operation = 5*ceil(max(h,w)/1024);
para.min_size = round(0.001*h*w);

[para.p1,para.p2] = CM_detection(im,para);
if show_match
    draw_match(img_rgb,para.p1,para.p2);
end
map = CM_localization(im,para);
figure;
imshow(map)