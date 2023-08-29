clear; clc;
beta = 0.5;
escala = 1;%.0125;
sigma = 1;%sqrt(sqrt(.5));
cutoff = 0; % Numero de pontos cortados no inicio da curva
N=0; % se N = 0 --> usa a distancia em blocos maxima
     % caso contrario, usa a distancia em blocos especificada 
Jump = 1;
band = 0;
YLIM =1.2;
TOL = 0.5e-1;
L=0.5;
%
for I=1:1
    arqq = num2str(I);
    arqq = ['Entrada arquivo de dados n.' arqq' ' (X)'];
    clear base_aux base_name
    home1 = './out/'
    home_fig = '../figuras/';
    home = [home1 'gs_e*x.dat']
    [FILENAME, PATHNAME] = uigetfile(home,...
        arqq);
    line_file=sprintf('%s%s', PATHNAME,FILENAME);
    arquivo1 = line_file
    s_a=size(arquivo1); s_a=s_a(1,2);
    arquivo = arquivo1(1:1:s_a-5);
    arquivo2 = [arquivo 'y.dat']
    base_name = [];
    for i=s_a:-1:1
        aa = arquivo1(1,i);
        if aa == '.';
            i=i-2;
            aa = arquivo1(1,i);
             while aa ~= '/'
                 if aa == '.'
                     aa = '_';
                 end
                 base_name = [aa base_name];
                 i=i-1;
                 aa = arquivo1(1,i);
             end
             break
        end
    end

    base_aux = [home_fig 'te_'];
    base_name=[base_aux base_name];

    base = 1;
    base_char = num2str(base,3);
    if I==1
        arq1 = load(arquivo1);
        arq2 = load(arquivo2);
    end
    if I==2
        arq3 = load(arquivo1);
        arq4 = load(arquivo2);
    end
    if I==3
        arq5 = load(arquivo1);
        arq6 = load(arquivo2);
    end
end
    for nn=1:2
        if(band==1)
            N=0;
        end
        for I=1:1
            if I==1
                if nn==1
                    dados1 = arq1;
                    base_name2=[base_name '_Yx'];
                end
                if nn==2
                    dados1 = arq2;
                    base_name2=[base_name '_Yy'];
                end
            end
            if I==2
                if nn==1
                    dados1 = arq3;
                    base_name2=[base_name '_Yx'];
                end
                if nn==2
                    dados1 = arq4;
                    base_name2=[base_name '_Yy'];
                end
            end
            if I==3
                if nn==1
                    dados1 = arq5;
                    base_name2=[base_name '_Yx'];
                end
                if nn==2
                    dados1 = arq6;
                    base_name2=[base_name '_Yy'];
                end
            end

            s1 = size(dados1);

            %D1 = dados1(1:3,:);

            d1 = dados1(1+cutoff:end,:);
            d1 = [escala*(d1(:,1)) sigma*d1(:,2)];
            sd1=size(d1);
            if N==0
                N=sd1(1,1);
                band=1;
                for i=1:size(d1,1)
                    if d1(i,2)<TOL
                        END=i-1;
                        break;
                    else
                        END=i;
                    end
                end
                if(END<N)
                    N=END
                end
            end
            alc = N

            l1 = (d1(1:N,:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %/%/%/%/%/%/%/%/%//%/%/%/%/%/%/%/%/%/%/%//%/%/%/
                ss=size(l1);
                ld1 = log(l1(1:alc,:));
                %ld2 = (l1(1:alc,:));
                xx = [min(l1(:,1)):(l1(2,1)-l1(1,1))*0.1:max(l1(:,1))]';

                d=(l1);
                d_x = (d(:,1));
                d_y = log(d(:,2));
                aux=corrcoef([d_x d_y]);
                coefr=aux(1,2)*aux(1,2);
            
                RL = polyfit(d_x,d_y,2);
                CR=num2str(coefr,'%1.2f');
                aa=num2str(sqrt(-1/(RL(1,1)*2)),'%1.2f');
                if(RL(1,2)>=0.0)
                    bb=num2str((RL(1,2)),'%1.2f');
                    bb=[bb 'r'];
                    sgn='+';
                else
                    bb=num2str(-(RL(1,2)),'%1.2f');
                    bb=[bb 'r'];
                    sgn='-';
                end
                if(abs(RL(1,2))<1e-10)
                    bb='';
                    sgn='';
                end
                cc=num2str(exp(RL(1,3)),'%1.2f');
%                pp=exp(RL(1,3)+RL(1,2)*xx + RL(1,1)*xx.^2);
                pp=exp(-((xx).^2)./(2*L^2));
                %yy='   $\langle\xi(\vec{x}) \xi( \vec{x}+\vec{r} )\rangle =';
                yy='   $\mathcal{C}_{Y}( {r} ) =';
                Xx=' \ e';
            if I==1
                for i=1:size(d1,1)
                    if d1(i,2)<-100
                        END=i-1;
                        break;
                    else
                        END=i;
                    end
                end
%
                x1 = dados1(1:Jump:END,1);
                y1 = dados1(1:Jump:END,2);
                d1 = [x1 y1];
                if nn == 1
                    % encontrar s maximos
                    marq1 = max(d1);
                    marq2 = max(d1);
                    miarq1 = min(d1);
                    miarq2 = min(d1);

                    if marq1(1)>marq2(1)
                        max_xx = marq1(1);
                    else
                        max_xx = marq2(1);
                    end
                    if miarq1(1)<miarq2(1)
                        min_xx = miarq1(1);
                    else
                        min_xx = miarq2(1);
                    end
                    if marq1(2)>marq2(2)
                        max_pp = marq1(2);
                    else
                        max_pp = marq2(2);
                    end
                    if miarq1(2)<miarq2(2)
                        min_pp = miarq1(2);
                    else
                        min_pp = miarq2(2);
                    end
%                     x_ini = min_xx*0.9*0;
%                     x_lim = max_xx*1.25;
%                     y_ini = min_pp*0.75;
%                     y_lim = max_pp*1.25;
                    x_ini = min_xx*0.9*0;
                    x_lim = max_xx*1.25;
                    y_ini = 0;
                    y_lim = YLIM;
                end
                xx1 = xx;
                pp1 = pp;
                aa = ['\mathbf{' aa '}'];
                bb = ['\mathbf{' bb '}'];
                cc = ['\mathbf{' cc '}'];
                CR = ['\mathbf{' CR '}']; 
                equation1=['$\begin{array}{rcl}\mathcal{\ C}_{Y} (\ {r}\ )\! &=&' cc  Xx '^{^{- \ \ \frac{1}{2}\frac{r}{' aa '^2} ' sgn '\ \ ' bb ' }} \\ R^{\ \, 2} &=& ' CR '\end{array}$'];
                equation12=['$\quad R^{\ \!2} = \ $' CR];
            end
            if I==2
                x2 = d1(:,1);
                y2 = d1(:,2);
                xx2 = xx;
                pp2 = pp;
                equation2=['$\mathcal{C}_{Y_{\!\!2}}\ ( {r} ) =' bb  Xx '^{^{ \-  \, \frac{r}{' aa '}}} \\ \quad R^{\ \!2} = \ ' CR '$'];
                equation22=['$\quad R^{\ \!2} = \ $' CR];
            end
            if I==3
                x3 = d1(:,1);
                y3 = d1(:,2);
                xx3 = xx;
                pp3 = pp;
                equation3=['$\mathcal{C}_{Y_{\!\!3}}\ ( {r} ) =' bb  Xx '^{^{ \ -  \, \frac{r}{' aa '}}} \\ \quad R^{\ \!2} = \ ' CR '$'];
                equation32=['$\quad R^{\ \!2} = \ $' CR];
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
figure1 = figure(...
  'PaperUnits','centimeters',...
  'PaperPosition',[0.9 0.35 16. 15.3],...
  'PaperSize',[17.3 15.7],...
  'PaperType','<custom>');
 
%% Create axes
axes1 = axes(...
  'DataAspectRatio',[1/(y_lim-y_ini) 1/(x_lim-x_ini) 1],...
  'FontName','Times',...
  'FontSize',22,...%'Position',[0.075 0.135 0.9 0.83],...
  'Parent',figure1);
ylim(axes1,[y_ini y_lim]);
xlim(axes1,[x_ini x_lim]);
xlabel(axes1,'${r}$','FontSize',26,'Interpreter','latex');
ylabel(axes1,'$\mathcal{\ C}_{Y}$','FontSize',26,'Interpreter','latex');
set(get(gca,'YLabel'),'Rotation', 90.0);
hold(axes1,'all');

            %% Create plot
            plot1 = plot(...
              x1,y1,...
              'Color',[0 0 0],...
              'LineStyle','none',...
              'Marker','o',...
              'MarkerSize',8,...
              'Parent',axes1);
            hold on;
            plot2 = plot(...
              xx1,pp1,...
              'Color',[1 0 0],...
              'LineStyle','-',...
              'MarkerSize',6,...
              'LineWidth',2,...
              'Parent',axes1);
            %% Create plot
            equation1='$\mathbf{exact}$';
            %% Create legend
            legend1 = legend(...
              axes1,{...
              '$\mathbf{computed}$',equation1},...
              'FontName','times',...
              'FontSize',18,...
              'Position',[0.3 0.65 0.9701 0.1849],...
              'Interpreter','latex');
%            legend(legend1,'box', 'off');
            box('on');
            base2 = ['$b = ' base_char '$'];
            %% Create textbox
          
              clear figure1 axes1
        %base_aux = 'exp_'
        if nn==2
            base_name2= base_name2(1:size(base_name2,2)-1);
            base_name2 = [base_name2 'y']
        else
            base_name2
        end
        %print('-djpeg99',base_name2);
        print('-depsc',base_name2);
        %/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %alc
    end
%clear;
