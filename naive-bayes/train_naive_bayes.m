function p_hat = train_naive_bayes(spam_data, nonspam_data)
  cols = size(spam_data, 2);
  p_hat = zeros(2, cols);
  p_hat(1, :) = sum(spam_data, 1)' / size(spam_data, 1);
  p_hat(2, :) = sum(nonspam_data, 1)' / size(nonspam_data, 1);
end
