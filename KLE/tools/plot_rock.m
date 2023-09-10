function plot_rock(rock,g,flag,titlen,color,lim,vw,n)
    fig = figure(n);
    Lx0 = min(g.nodes.coords(:,1));
    Lx  = max(g.nodes.coords(:,1));
    Ly0 = min(g.nodes.coords(:,2));
    Ly  = max(g.nodes.coords(:,2));
    Lz0 = min(g.nodes.coords(:,3));
    Lz  = max(g.nodes.coords(:,3));
    dx  = (Lx - Lx0)/4;
    dy  = (Ly - Ly0)/4;
    dz  = (Lz - Lz0)/1;
    if flag == 'Y'
        beta = 1.0; rho = 1.0;
        perm = reverseKlog(rock,beta,rho);
    else
        perm = rock;
    end
    clear rock
    if norm(lim)< 1e-17
        lim = [min(min(perm)) max(max(perm))];
        if(abs(lim(1)-lim(2))<1.e-06)
            lim(1) = lim(1)*0.8; lim(2) = lim(2)*1.2;
            lim = sort(lim);
        end
    end
    axes('DataAspectRatio',[1 1 1],'FontName','Times','FontSize',12,...
        'PlotBoxAspectRatio',g.cartDims,'TickDir','both',...
        'TickLabelInterpreter','latex','XTick',[Lx0:dx:Lx],...
        'YTick',[Ly0:dy:Ly],'ZDir','reverse','ZTick',[Lz0:dz:Lz],...
        'TickLength',[0.0125 0.025],'XMinorTick','on','YMinorTick','on',...
        'ZDir','reverse','ZMinorTick','on');

    plotCellData(g,perm,'EdgeColor', color,'FaceAlpha',.95);
    colorbar('horiz','southoutside','TickLabelInterpreter','latex','FontSize',11);
    axis equal tight; view(vw); box 'on';
    caxis([lim(1) lim(2)]); colormap(jet(55));
    title(titlen,'FontSize',14,'Interpreter','latex');

    % Create labels
    zlabel('$z$','FontSize',14,'Interpreter','latex');
    ylabel('$y$','FontSize',14,'Interpreter','latex');
    xlabel('$x$','FontSize',14,'Interpreter','latex');
%     [hc,hh]=colorbarHist(rock,lim,'South', 80);
%     pos=get(hc,'Position'); set(hc,'Position',pos - [.02 0 .2 -.01],'FontSize',12);
%     pos=get(hh,'Position'); set(hh,'Position',pos - [.02 0.02 .2 -.01]);
%     %set(gca,'Position',[.13 .2 .775 .775])

end