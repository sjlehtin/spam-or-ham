function g=classify_with_naive_bayes(unknown, p_hat, p_c)

  p_hat(p_hat == 0) = 0.00000000000000001;

  g1 = sum( unknown * log(p_hat(1, :)') ...
	  + (1 - unknown) * log(1 - p_hat(1, :)'), 2) ...
	  + log(p_c(1, :))

  g2 = sum( unknown * log(p_hat(2, :)') ...
	  + (1 - unknown) * log(1 - p_hat(2, :)'), 2) ...
	  + log(p_c(2, :))

  g = g1 > g2;
end
