function[X,Lx,Ly,Lz,nx,ny,nz]= leitura3D(name)
    TOL = 1e-6;
    outfile2 = fopen(name, 'r');
    A=fscanf(outfile2,'%f');
    Lx=A(1);
    Ly=A(2);
    Lz=A(3);
    nx=int64(A(4));
    ny=int64(A(5));
    nz=int64(A(6));
    ntipo=A(7);
    beta=A(8);
    lixo=A(9);
    lixo=A(10);
    Xi=zeros(nx*ny*nz,1);
    m=0;
    k=10;
    n=size(A);
        for z=1:nz
            k=k+1;
            if(int64(A(k))~=z-1)
                disp('Problema na leitura z')
                break
            end
            for i=1:1:ny
                k=k+1;
                if(int64(A(k))~=i-1)
                    disp('Problema na leitura y')
                    break
                end
                k=k+1;
                for j=1:nx
                    m=m+1;
                    Xi(m)=A(k);
                    k=k+1;
                end
                if(int64(A(k))~=192837465)
                    disp('Problema na leitura z')
                    break
                end
            end
        end
        fclose(outfile2);
        X = zeros(nx*ny*nz,1);
        M = nx*ny;
        N = nx*ny*nz;
        for z=0:nz-1
            X(N+1-(z+1)*M:N-z*M) = Xi(z*M+1:(z+1)*M);
        end
end
%
