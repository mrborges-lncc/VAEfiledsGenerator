function [K] = load_perm(G,meshInfo,namex,namey,namez,depth,nini,nD)
    dp = depth;
    TOL= 1.0e-07;
    n  = G.cartDims;
    d  = G.griddim;
    Lx = meshInfo.dims(1); Ly = meshInfo.dims(2); Lz = meshInfo.dims(3);
    L  = min(G.nodes.coords);
    Lx0 = L(1); Ly0 = L(2); 
    if(d == 2)
        Lz0 = 0.0;
    else
        Lz0 = L(3);
    end
    nx = int16(meshInfo.fine(1));
    ny = int16(meshInfo.fine(2));
    nz = int16(meshInfo.fine(3));
    dx = (Lx-Lx0)/double(nx);
    dy = (Ly-Ly0)/double(ny);
    dz = (Lz-Lz0)/double(nz);
    [yx, Llx, Lly, Llz, nnx, nny, nnz] = perm_reader(namex,nini,nD);
    if(nD == '2D') Llz = Lz; dp = 0.0; end;
    if(abs(Llx-Lx)>TOL || abs(Lly-Ly)>TOL || abs(Llz-Lz+dp)>TOL )
        error('Wrong dimention (1)'); 
    end
    [yy, Llx, Lly, Llz, nnx, nny, nnz] = perm_reader(namey,nini,nD);
    if(nD == '2D') Llz = Lz; dp = 0.0; end;
    if(abs(Llx-Lx)>TOL || abs(Lly-Ly)>TOL || abs(Llz-Lz+dp)>TOL )
        error('Wrong dimention (2)');
    end
    if(nD == '3D')
        [yz, Llx, Lly, Llz, nnx, nny, nnz] = perm_reader(namez,nini,nD);
        if(nD == '2D') Llz = Lz; dp = 0.0; end;
        if(abs(Llx-Lx)>TOL || abs(Lly-Ly)>TOL || abs(Llz-Lz+dp)>TOL )
            error('Wrong dimention (3)');
        end
    else
        yz = zeros(nnx*nny*nnz,1);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Log-permeability %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    K = [yx yy yz];
    clear yx yy yz
    if(nx == nnx && ny == nny && nz==nnz)
        fprintf('\nFile number %d OK\n',nini);
    else
        if(nnx > nx || nny > ny || nnz > nz )
            error('Upscaling required')
        else
            Geo  = cartGrid([nnx nny nnz],[Lx Ly (Lz-Lz0)]*meter^3);
            Geo.nodes.coords(:, 3) = depth + Geo.nodes.coords(:, 3)*meter;
            Geo.nodes.coords(:, 2) = Geo.nodes.coords(:, 2)*meter;
            Geo.nodes.coords(:, 1) = Geo.nodes.coords(:, 1)*meter;
            Geo  = computeGeometry(Geo);
            K = mappingGeo(Geo,G,K,(Lx-Lx0)/double(nnx),...
                (Ly-Ly0)/double(nny),(Lz-Lz0)/double(nnz));
        end
        clear Geo
    end
    K = K(:,1:d);
    return
end

