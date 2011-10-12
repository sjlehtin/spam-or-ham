function g=classify_with_naive_bayes(unknown, p_hat, p_c)

  p_hat(p_hat == 0) = 0.00000000000000000000000000001;

  g1 = sum( unknown * log(p_hat(1, :)') ...
	  + (1 - unknown) * log(1 - p_hat(1, :)'), 2) ...
	  + log(p_c(1, :));

  g2 = sum( unknown * log(p_hat(2, :)') ...
	  + (1 - unknown) * log(1 - p_hat(2, :)'), 2) ...
	  + log(p_c(2, :));

  p1 = exp(g1);
  p2 = exp(g2);
  s = p1 + p2;
  p1 = p1 ./ s;
  p2 = p2 ./ s;

  g = g1 > g2;
end
