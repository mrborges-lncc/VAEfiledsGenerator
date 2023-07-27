function [pnode] = InfoConditioning(TIPOINPUT, vet, n_dados, ...
    Lx, Ly, Lz, nx, ny, nz)
    if(TIPOINPUT == 1)
        hx = Lx / double(nx);
        hy = Ly / double(ny);
        hz = Lz / double(nz);
        NT = n_dados;
        A  = zeros(n_dados,n_dados);
        vet= vet-1e-6;
        pnode=zeros(n_dados,1);
        for nn=1:n_dados
            k=0;
            for l=1:nz
                zf=l*hz;
                zi=zf-hz;
                if(((vet(nn,3)<zf)&&(vet(nn,3)>zi))||(nz==1))
                    for j=ny:-1:1
                        yi = (ny-j)*hy;
                        yf = yi+hy;
                        if((vet(nn,2)<yf)&&(vet(nn,2)>yi))
                            for i=1:nx
                                xf = i*hx;
                                xi = xf - hx;
                                if((vet(nn,1)<xf)&&(vet(nn,1)>xi))
                                    k=k+1;
                                    pnode(nn)=k;
                                else
                                    k=k+1;
                                end
                            end
                        else
                            k=k+nx;
                        end
                    end
                else
                    k=k+nx*ny;
                end
            end
        end
    else
        pnode = [];
    end    
    return
end
