function [name_autovet, name_autoval, name_mutilde, ...
    name_MMat] = fnames(homeT,ntipo,tipo,beta,eta1,eta2,eta3,...
    Lx,Ly,Lz,NX,NY,NZ,M)
    if(ntipo==2)
        coefic=['_' num2str(beta)];
    else
        coefic=['_' num2str(eta1) 'x' num2str(eta2) 'x' num2str(eta3)];
    end
    name_aux = ['_' tipo num2str(ntipo,5) '_'...
        num2str(Lx) 'x' num2str(Ly) 'x' num2str(Lz) '_'...
        num2str(NX) 'x' num2str(NY) 'x' num2str(NZ) ...
        coefic '_M' num2str(M) '.dat'];
    name_autovet = [homeT 'avet' name_aux];
    name_autoval = [homeT 'aval' name_aux];
    name_mutilde = [homeT 'mutilde' name_aux];
    name_MMat    = [homeT 'MM' name_aux];
    fprintf('\n#########################################')
    fprintf('#########################################')
    fprintf('\nFile name of eigen-vectors: %s',name_autovet)
    fprintf('\nFile name of eigen-values.: %s',name_autoval)
    fprintf('\nFile name of mu-tilde.....: %s',name_mutilde)
    fprintf('\nFile name of M-matrix.....: %s',name_MMat)
    fprintf('\n#########################################')
    fprintf('#########################################\n')
    return
end