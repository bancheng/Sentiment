f1 = fopen('word_list_unstem4.txt','r');
word_str = fgetl(f1);
word_list = regexp(word_str, ' ', 'split');
fclose(f1);
%% 用6000个词汇来表示5330条负极的影评句子
f2 = fopen('token_data_neg.txt','r');
matrix_doc_neg = zeros(5330, 4000);
for l = 1 : 5330
    sentance = fgetl(f2);
    sentance_list = regexp(sentance, ' ', 'split');
    for i = 1 : length(sentance_list)
        for j = 1 : length(word_list)
            if(strcmp(sentance_list{i}, word_list{j}))
                matrix_doc_neg(l, j) = 1;
                break;
            end
        end
    end
        
end
fclose(f2);
save matrix2_doc_neg matrix_doc_neg

%% 用6000个词汇来表示5329条正极的影评句子
f3 = fopen('token_data_pos.txt','r');
matrix_doc_pos = zeros(5329, 4000);
for l = 1 : 5329
    sentance = fgetl(f3);
    sentance_list = regexp(sentance, ' ', 'split');
    for i = 1 : length(sentance_list)
        for j = 1 : length(word_list)
            if(strcmp(sentance_list{i}, word_list{j}))
                matrix_doc_pos(l, j) = 1;
                break;
            end
        end
    end
        
end
fclose(f3);
save matrix2_doc_pos matrix_doc_pos