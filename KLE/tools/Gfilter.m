function [x] = Gfilter(xi,num_elem,epsilon)
    x = zeros(num_elem,1);
    delta = norminv(epsilon);
    x(xi<delta) = 1;
end