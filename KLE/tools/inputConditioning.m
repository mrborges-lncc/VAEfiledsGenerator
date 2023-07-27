function [vet, dados, n_dados] = inputConditioning(file_input_cond,TIPOINPUT)
    if(TIPOINPUT == 1)
        inp = load(file_input_cond);
        vet = inp(:,1:3);
        dados=inp(:,4);
        clear inp
    else
        vet=[];
        dados=[];
    end
    n_dados = size(vet,1);
%     if(TIPOINPUT == 1)
%         NT=n_dados;
%         A=zeros(n_dados,n_dados);
%         vet=vet+1e-6;
%         pnode=[];
%         for nn=1:n_dados
%             k=0;
%             for l=1:nz
%                 zf=l*hz;
%                 zi=zf-hz;
%                 if(((vet(nn,3)<zf)&&(vet(nn,3)>zi))||(nz==1))
%                     for j=ny:-1:1
%                         yi = (ny-j)*hy;
%                         yf = yi+hy;
%                         if((vet(nn,2)<yf)&&(vet(nn,2)>yi))
%                             for i=1:nx
%                                 xf = i*hx;
%                                 xi = xf - hx;
%                                 if((vet(nn,1)<xf)&&(vet(nn,1)>xi))
%                                     k=k+1;
%                                     pnode(nn)=k;
%                                 else
%                                     k=k+1;
%                                 end
%                             end
%                         else
%                             k=k+nx;
%                         end
%                     end
%                 else
%                     k=k+nx*ny;
%                 end
%             end
%         end
%         pnode = pnode';
%     %%
%     % *** vetor de posiccoes em relaccao aos nos da malha *********************
%     % *** apenas para nodes ***************************************************
%         pt = zeros(num_elem-n_dados,1);
%         k=0;
%         for i=1:num_elem
%             sgn = 1;
%             for j=1:n_dados
%                 if (i == pnode(j))
%                     sgn = -1;
%                     break;
%                 end
%             end
%             if sgn>0
%                 k=k+1;
%                 pt(k)=i;
%             end
%         end
%     end    
    return
end
