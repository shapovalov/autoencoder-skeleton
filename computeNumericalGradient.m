function numgrad = computeNumericalGradient(J, theta)
	% numgrad = computeNumericalGradient(J, theta)
	% theta: a vector of parameters
	% J: a function that outputs a real-number. Calling y = J(theta) will return the
	% function value at theta.
	EPSILON = 1e-4;

	variables_count = length(theta);
	ei = zeros(size(theta));
	ei(1) = 1;
	for i = 1:variables_count
		numgrad(i) = J(theta + EPSILON * ei) - J(theta - EPSILON * ei);
		numgrad(i) = numgrad(i) / (2 * EPSILON);
		if (i < variables_count)
			ei(i) = 0;
			ei(i + 1) = 1;
		end
	end
end
