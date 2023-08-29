function[Xi]= leitura3D(Lx,Ly,Lz,nx,ny,nz,ntipo,beta,nr,home,name,tipo_prt)
    if tipo_prt == 1
        nfile= num2str(nr,5);
        bb2 = [home name nfile '.dat'];
        outfile2 = fopen(bb2, 'r');
        A=fscanf(outfile2,'%f');
        Lx=A(1);
        Ly=A(2);
        Lz=A(3);
        nx=A(4);
        ny=A(5);
        nz=A(6);
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
            if(A(k)~=z-1)
                disp('Problema na leitura z')
                break
            end
            for i=ny:-1:1
                k=k+1;
                if(A(k)~=ny-i)
                    disp('Problema na leitura y')
                    break
                end
            k=k+1;
            for j=1:nx
                m=m+1;
                Xi(m)=A(k);
                k=k+1;
            end
            if(A(k)~=192837465)
                disp('Problema na leitura z')
                break
            end
            end
        end
        fclose(outfile2);
    end
    if tipo_prt == 0
        nfile= num2str(nr,5);
        bb2 = [home name nfile '.dat'];
        Xi = load(bb2);
    end
    fprintf('Reading file %s\n',bb2)
    return
end
%
