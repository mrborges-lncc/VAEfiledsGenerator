function [nx,ny,nz,NX,NY,NZ,M,tEINT] = preInterpolation(interpol,...
    nx,ny,nz,NX,NY,NZ,M)
    tEINT = 0.0;
    if(interpol~=1)
        NX=nx;
        NY=ny;
        NZ=nz;
        tEINT=0.0;
    else
        if(nx==NX)
            nx = nx;
        else
            nx = nx+1;
        end
        if(ny==NY)
            ny = ny;
        else
            ny = ny+1;
        end
        if(nz==NZ)
            nz = nz;
        else
            nz = nz+1;
        end
    %
        if(NX<nx)
            fprintf('\n\n#######################################\n')
            error('Ploblem in the interpolation: NX < nx');
            return
        end
        if(NY<ny)
            fprintf('\n\n#######################################\n')
            error('Ploblem in the interpolation: NY < ny');
            return
        end
        if(NZ<nz)
            fprintf('\n\n#######################################\n')
            error('Ploblem in the interpolation: NZ < nz');
            return
        end
    end
    if(M>nx*ny*nz)
        fprintf('PROBLEM IN M SIZE\n');
        fprintf('ACTUAL M: %d\n',M);
        fprintf('MAXIMUM M: %d\n',nx*ny*nz);
        M = input('Enter new M value: ');
    end
    if(M<=0)
        M=nx*ny*nz;
    end
end

