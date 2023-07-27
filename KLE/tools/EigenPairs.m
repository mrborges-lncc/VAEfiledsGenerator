function [lambda, phi] =  EigenPairs(C)
    C = (C + conj(C)') / 2;
    [phi, D] = eig(C,'vector');
    [lambda, ind] = sort(D,'descend');
    phi = phi(:,ind);
    clear D ind C
return