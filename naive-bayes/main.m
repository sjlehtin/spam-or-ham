function returnvalue = main()

  SPAM_ROWS = 500;
  NON_SPAM_ROWS = 500;
  KNOWN_ROW_COUNT = SPAM_ROWS + NON_SPAM_ROWS;

  data = load('T-61_3050_data.txt');
  
  classified_indices = data(KNOWN_ROW_COUNT + 1 : end, [1]);

  known_data = data(1 : KNOWN_ROW_COUNT, :);
  known_data(:, [1:2]) = zeros(KNOWN_ROW_COUNT, 2);

  data = data(:, not(sum(known_data, 1) == 0));

  spam_data = data(1 : SPAM_ROWS, 3 : end);
  nonspam_data = data(SPAM_ROWS + 1 : KNOWN_ROW_COUNT, 3 : end);
  unkown_data = data(KNOWN_ROW_COUNT + 1 : end, 3 : end);
  clear data;

  p_hat = train_naive_bayes(spam_data, nonspam_data);
	  
  classified = classify_with_naive_bayes(unkown_data, p_hat, ...
	  [SPAM_ROWS / KNOWN_ROW_COUNT; NON_SPAM_ROWS / KNOWN_ROW_COUNT]);


  returnvalue = [classified_indices, classified];
