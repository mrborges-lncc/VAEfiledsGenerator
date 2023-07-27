function [C, sigma, num_elem] = bildMatrix(Lx,Ly,Lz,nx,ny,nz,...
    eta1,eta2,eta3,beta,nu,ntipo,varY,interpolacao)
    %** Parameters adjust *************************************************
    sigma    = sqrt(varY);
    num_elem = nx*ny*nz;
    if interpolacao == 1
        hx= Lx/double(nx-1);
        hy= Ly/double(ny-1);
        hz= Lz/double(nz-1);
        x = [0 : hx : (nx-1) * hx];
        y = [0 : hy : (ny-1) * hy];
        z = [0 : hz : (nz-1) * hz];
    else
        hx= Lx/double(nx);
        hy= Ly/double(ny);
        hz= Lz/double(nz);
        x = [hx/2 : hx : nx * hx];
        y = [hy/2 : hy : ny * hy];
        z = [hz/2 : hz : nz * hz];
    end
    if(ntipo==2)
        alpha  = 1.0;
        cutoff = Lx/double(nx);
        ft     = sqrt(2.0)*(varY);
        fat    = (varY)*(hx^beta)*((hx/cutoff)^-beta);
    else
        fat = 1.0;
        ft  = varY;
    end
    [Y,X,Z] = meshgrid(single(y),single(x),single(z));
    C       = single(zeros(num_elem,num_elem));
    ephilon = hx;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ntipo == 1
        coord  = [reshape(X,[nx * ny * nz, 1]) / eta1 ...
            reshape(Y,[nx * ny * nz, 1]) / eta2 ...
            reshape(Z,[nx * ny * nz, 1]) / eta3];
        for i = 1 : num_elem
            C(:,i) = varY * exp(-sqrt(sum((coord(i,:) - coord).^2,2)));
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ntipo == 3
        c = sqrt(2.0);
        coord  = [reshape(X,[nx * ny * nz, 1]) / (c*eta1) ...
            reshape(Y,[nx * ny * nz, 1]) / (c*eta2) ...
            reshape(Z,[nx * ny * nz, 1]) / (c*eta3)];
        for i = 1 : num_elem
            C(:,i) = varY * exp(-(sum((coord(i,:) - coord).^2,2)));
        end
    end
    wP=1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ntipo == 2
        aux1 = (fat*alpha^(beta)*sigma^2*(hx/4)^2*wP') * ephilon^beta;
        aux  = alpha^(beta)*sigma^2*hx^2*sqrt(2);
        coord = [reshape(X,[nx * ny * nz, 1]) ...
            reshape(Y,[nx * ny * nz, 1]) ...
            reshape(Z,[nx * ny * nz, 1])];
        for i = 1 : num_elem
            C(:,i) = aux1 * sqrt(sum((coord(i,:) - coord).^2,2)).^(-beta);
            C(i,i) = aux;
        end        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ntipo==4
        coord = [reshape(X,[nx * ny * nz, 1]) ...
            reshape(Y,[nx * ny * nz, 1]) ...
            reshape(Z,[nx * ny * nz, 1])].';
        l1=eta1*eta1;
        l2=eta2*eta2;
        l3=eta3*eta3;
        for ei = 1:num_elem
            zi = coord(:,ei)';
            xv = [zi(1),zi(2),zi(3)];
            for ej = ei:num_elem
                zj = coord(:,ej)';
                yv = [zj(1),zj(2),zj(3)];
                C(ei,ej) = matern(xv,yv,varY,l1,l2,l3,nu);
                C(ej,ei) = C(ei,ej);
            end
        end
    end
    clear xv yv wP aux coord zi zj X Y Z
    return
end