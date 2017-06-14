%5330-neg 1 : 5330
%5329-pos 1 : 5329
matrix_doc_neg = load('matrix2_doc_neg.mat');
matrix_doc_pos = load('matrix2_doc_pos.mat');
matrix_doc_neg = matrix_doc_neg.matrix_doc_neg;
matrix_doc_pos = matrix_doc_pos.matrix_doc_pos;
%% 处理掉全为0或者全为1的feature, 剩下5967个features
matrix_neg = matrix_doc_neg(:, 1:4000);
matrix_pos = matrix_doc_pos(:, 1:4000);
for j = 1:4000
    if(~any(matrix_doc_neg(:, j)) && ~any(matrix_doc_pos(:, j)))
        matrix_neg(:, j) = [];
        matrix_pos(:, j) = [];
    end
end
matrix_doc_neg = matrix_neg;
matrix_doc_pos = matrix_pos;
clear matrix_neg matrix_pos;
%% partitioning，按7：3的比例分为train_set 和 test_set，3700 * 2 个training set， 1600 * 2个test set
k1 = randperm(5330);
k2 = randperm(5329);

train_neg = matrix_doc_neg(k1(1 : 3700), :);
test_neg = matrix_doc_neg(k1(3731 : end), :);
train_pos = matrix_doc_pos(k2(1 : 3700), :);
test_pos = matrix_doc_pos(k2(3730 : end), :);

%training set, 标记分两类，第一类为positive， 第二类为negative, 数量相等
train_set = zeros(3700 * 2, 3977);
train_set(1 : 3700,: ) = train_neg;
train_set(3701 : end, :) = train_pos;
k3 = randperm(3700 * 2);
train_x = train_set(k3(1 : end), :);
train_y = zeros(3700 * 2, 2);
y1 = (k3 > 3700)';
for i = 1 : 3700 * 2
    if(y1(i,1) == 1)
        train_y(i, 1) = 1;
    else
        train_y(i, 2) = 1;
    end
end

%test set，标记分两类，第一类为positive， 第二类为negative, 数量相等
test_set = zeros(1600 * 2, 3977);
test_set(1 : 1600, :) = test_neg;
test_set(1601 : end, :) = test_pos;

k4 = randperm(1600 * 2);
test_x = test_set(k4(1 : end), :);
test_y = zeros(1600 * 2, 2);
y2 = (k4 > 1600)';
for i = 1 : 1600 * 2
    if(y2(i,1) == 1)
        test_y(i, 1) = 1;
    else
        test_y(i, 2) = 1;
    end
end

save finaldataset_unstem4 train_x train_y test_x test_y


