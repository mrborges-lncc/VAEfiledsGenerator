function [newT,num_elem] = interpMatrixT(T,M,interpolacao,Lx,Ly,Lz,...
    nx,ny,nz,NX,NY,NZ)
    if(interpolacao==1)
        if(NX==nx)
            dx = Lx/double(nx);
            x1 = dx/2:dx:Lx;
        else
            dx = Lx/double(nx-1);
            x1 = 0:dx:Lx;
        end
        if(NY==ny)
            dy = Ly/double(ny);
            y1 = dy/2:dy:Ly;
        else
            dy = Ly/double(ny-1);
            y1 = 0:dy:Ly;
        end
        if(NZ==nz)
            dz = Lz/double(nz);
            z1 = dz/2:dz:Lz;
        else
            dz = Lz/double(nz-1);
            z1 = 0:dz:Lz;
        end
    %
        [X1,Y1,Z1] = meshgrid(single(x1),single(y1),single(z1));
    %
        dx = Lx/double(NX);
        dy = Ly/double(NY);
        dz = Lz/double(NZ);
        x1 = dx/2:dx:Lx;
        y1 = dy/2:dy:Ly;
        z1 = dz/2:dz:Lz;
        [X2,Y2,Z2] = meshgrid(single(x1),single(y1),single(z1));
        clear x1 y1 z1
        newT = single(zeros(NX*NY*NZ,M));
        for m=1:M
            vect=zeros(ny,nx,nz);
            k=0;
            for l=1:nz
                for j=1:1:ny
                    for i=1:nx
                        k=k+1;
                        vect(j,i,l)=T(k,m);
                    end
                end
            end
            if(nz==1)
                if(ny==1)
                    vect2=interp1(X1,vect,X2,'spline');
                    k=0;
                    for i=1:NX
                        k=k+1;
                        newT(k,m)=vect2(i);    
                    end
                else
                    vect2=interp2(X1,Y1,vect,X2,Y2,'spline');
                    k=0;
                    for j=1:1:NY
                        for i=1:NX
                            k=k+1;
                            newT(k,m)=vect2(j,i);
                        end
                    end
                end
            else
                vect2=interp3(X1,Y1,Z1,vect,X2,Y2,Z2,'spline');
                k=0;
                for l=1:NZ
                    for j=1:1:NY
                        for i=1:NX
                            k=k+1;
                            newT(k,m)=vect2(j,i,l);
                        end
                    end
                end
            end
        end
    else
        newT = T;
    end
    clear vect vect2 X1 X2 Y1 Y2 Z1 Z2 x1 y1 z1
%%% RESCALING THE PROBLEM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nx=NX;
    ny=NY;
    nz=NZ;
    num_elem = nx*ny*nz;
    return
end