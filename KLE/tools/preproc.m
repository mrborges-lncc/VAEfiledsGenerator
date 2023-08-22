function [dim, nD, fine_grid, coarse_grid, dims, mI] = preproc(...
    Lx,Ly,Lz,nx,ny,nz,nnx,nny,nnz)
    if mod(nx,nnx) ~= 0
        error('Coarse mesh should be multiple of fine mesh (mod(nx,nnx) ~= 0)')
    end
    if mod(ny,nny) ~= 0
        error('Coarse mesh should be multiple of fine mesh (mod(ny,nny) ~= 0)')
    end
    if mod(nz,nnz) ~= 0
        error('Coarse mesh should be multiple of fine mesh (mod(nz,nnz) ~= 0)')
    end
    if nz == 1
        dim = 2;
        nD  = '3D';
    else
        dim = 3;
        nD  = '3D';
    end
    dims        = [Lx Ly Lz];
    fine_grid   = int64([nx ny nz]);
    coarse_grid = int64([nnx nny nnz]);
    mI.dims     = dims;
    mI.fine     = fine_grid;
    mI.coarse   = coarse_grid;
    mI.fc       = fine_grid ./ coarse_grid;
%     for i = 1 : dim
%         if mI.fc(i) > 1 && mod(mI.fc(i),2)
%             mI.fc(i) = mI.fc(i) + 1;
%         end
%     end
    return
end
