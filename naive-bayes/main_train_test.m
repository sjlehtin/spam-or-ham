function [test_data_r, train_classified] = main()

  SPAM_ROWS = 500;
  NON_SPAM_ROWS = 500;
  KNOWN_ROW_COUNT = SPAM_ROWS + NON_SPAM_ROWS;

  data = load('T-61_3050_data.txt');
  
  known_data = data(1 : KNOWN_ROW_COUNT, :);
  known_data(:, [1:2]) = zeros(KNOWN_ROW_COUNT, 2);

  data = data(:, not(sum(known_data, 1) == 0));

  %%%
  known_data = data(1 : KNOWN_ROW_COUNT, :);
  idx = randperm(KNOWN_ROW_COUNT)';
  TRAIN_ROWS = 700;
  TEST_ROWS = KNOWN_ROW_COUNT - TRAIN_ROWS;

  t = idx(1:TRAIN_ROWS);
  train_data = [t <= 500, known_data(t, :)];

  t = idx(TRAIN_ROWS + 1 : KNOWN_ROW_COUNT);
  test_data = known_data(t, :);
  test_data_r = t <= 500;

  t = train_data(:, 2:end);
  train_data_spam = t(train_data(:, 1) == 1);
  train_data_nonspam = t(train_data(:, 1) == 0);

  train_p_hat = train_naive_bayes(train_data_spam, train_data_nonspam);
  train_classified = classify_with_naive_bayes(test_data, train_p_hat, ...
	  [sum(train_data(:, 1)) / TRAIN_ROWS;
	  sum(1 - train_data(:, 1)) / TRAIN_ROWS]);

  return
  %%%

  spam_data = data(1 : SPAM_ROWS, 3 : end);
  nonspam_data = data(SPAM_ROWS + 1 : KNOWN_ROW_COUNT, 3 : end);
  unkown_data = data(KNOWN_ROW_COUNT + 1 : end, 3 : end);
  clear data;

  p_hat = train_naive_bayes(spam_data, nonspam_data);
	  
  classified = classify_with_naive_bayes(unkown_data, p_hat, ...
	  [SPAM_ROWS / KNOWN_ROW_COUNT; NON_SPAM_ROWS / KNOWN_ROW_COUNT]);
