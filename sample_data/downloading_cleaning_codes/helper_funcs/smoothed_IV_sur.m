function [] = smoothed_IV_sur(T,M,IV,titl,x_label,y_label,z_label)
% choose bandwidth hT and hM
hT = median(abs(T-median(T)));    
surface.hT = hT;
hM = median(abs(M-median(M)));    
surface.hM = hM;


% choose grid step N 
TT = sort(T);     
MM = sort(M);
NT = histc(T,TT);
NM = histc(M,MM);
NT(NT==0) = [];   
NM(NM==0) = [];
nt=length(NT);    
nm=length(NM);
N=min(max(nt,nm),70);

% smoothe with Gaussian kernel
kerf = @(z)exp(-z.*z/2)/sqrt(2*pi);
surface.T = linspace(min(T),max(T),N);
surface.M = linspace(min(M),max(M),N);
surface.IV = nan(1,N);
for i=1:N
    for j=1:N
    z = kerf((surface.T(j)-T)/hT).*kerf((surface.M(i)-M)/hM); 
    surface.IV(i,j) = sum(z.*IV)/sum(z);
    end
end

% plot volatility surface
surf(surface.T, surface.M, surface.IV,'FaceAlpha',0.6)
colormap turbo;
axis tight; grid on;
title(titl,'Fontsize',24,'FontWeight','Bold','interpreter','latex'); 
% zlim([0 0.6]) 
xlabel(x_label,'Fontsize',20,'FontWeight','Bold','interpreter','latex');
ylabel(y_label,'Fontsize',20,'FontWeight','Bold','interpreter','latex');
zlabel(z_label,'Fontsize',20,'FontWeight','Bold','interpreter','latex');
set(gca,'Fontsize',16,'FontWeight','Bold','LineWidth',2);
