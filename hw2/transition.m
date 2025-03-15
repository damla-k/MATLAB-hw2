function x = transition(max_num, target_matrix)
  x = zeros(max_num, size(target_matrix, 2));
  for i = 1:size(target_matrix, 2)
    x(target_matrix(i)+1,i) = 1;
  end
end