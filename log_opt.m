function [opt_val, opt_point] = log_opt(z)
% Solves the optimization problem max sum_i log x_i + sum_i xi* z_i
% Subject to the constraint that x lies on standard simplex
% by using the direct KKT condition to solve the problem

est=-max(z)-1; % estimated value 
f_val=1;

while ((abs(f_val)>=10^-5)|| max(z+est)>0)
    f_val= sum(1./(z + est))+1;
    f_dash= -sum(1./(z+est).^2);
    est= est- f_val/f_dash;
end

opt_point= -1./(z+est);
opt_val= sum(log(opt_point)) + opt_point*z';

