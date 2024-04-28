clear all; close all
ntipo = 3;
eta = {[10.0, 10.0, 0.01], [15.0, 15.0, 0.01], [20.0, 20.0, 0.01],...
       [25.0, 25.0, 0.01], [30.0, 30.0, 0.01], [35.0, 35.0, 0.01],...
       [5.0, 5.0, 0.01]};
n = size(eta,2);
for i = 1 : n
    eta1 = eta{i}(1); eta2 = eta{i}(2); eta3 = eta{i}(3);
    functionKLEgenerator3D_conditioned(eta1, eta2, eta3, ntipo)
    close all
end
clear all