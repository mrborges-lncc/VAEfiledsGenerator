function T = TlambdaMatrix(phi,lambda,M,num_elem,nome)
    T = single(zeros(num_elem,M));
%     T=phi(:,1:M) * diag(sqrt(lambda(1:M)));
    T = phi(:,1:M) .* (sqrt(lambda(1:M))).';
    max_energy = sum(lambda);
    sgn90 = 1;
    sgn94 = 1;
    sgn96 = 1;
    for i = 1 : size(lambda,1)
        energy = sum(lambda(1:i)) * 100 / max_energy;
        if energy >= 90.0 && sgn90 == 1
            newM90 = i - 1;
            sgn90  = 0;
        end
        if energy >= 94.0 && sgn94 == 1
            newM94 = i - 1;
            sgn94  = 0;
        end
        if energy >= 96.0 && sgn96 == 1
            newM96 = i - 1;
            sgn96  = 0;
        end
        if energy >= 98.0
            newM = i - 1;
            break
        end
    end
    p = [90; 94; 96; 98];
    e = [newM90; newM94; newM96; newM];
    v = [p e];
    fprintf('\n####################################')
    fprintf('\n90 percent energy => M = %d',newM90)
    fprintf('\n94 percent energy => M = %d',newM94)
    fprintf('\n96 percent energy => M = %d',newM96)
    fprintf('\n98 percent energy => M = %d',newM)
    fprintf('\n####################################\n')
    save(nome,'v','-ascii');
    return
end
