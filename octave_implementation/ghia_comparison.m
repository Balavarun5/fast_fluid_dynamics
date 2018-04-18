function [] = plot()
y_paper = [0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,0.5,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766,1];
u100_paper = [0,-0.03717,-0.04192,-0.04775,-0.06434,-0.10150,-0.15662,-0.21090,-0.20581,-0.13641,0.00332,0.23151,0.68717,0.73722,0.78871,0.84123,1];
u400_paper = [0, -0.08186,-0.09266,-0.10338,-0.14612,-0.24299,-0.32726,-0.17119,-0.11477,0.02135,0.16256,0.29093,0.55892,0.61756,0.68439,0.75837,1];

%Change first two variables
N=[99,33,65];
dt = 0.01;

h100 = figure();
h400 = figure();

for i = 1:length(N)
    
    [u100,v100,~] = lid_driven_cavity(N(i), dt, 10, 1/100);
    [u400,v400,~] = lid_driven_cavity(N(i), dt, 20, 1/400);

    y_ffd = linspace(0,1,N(i));
    u100_ffd = u100(:,(N(i)-1)/2,end);
    u400_ffd = u400(:,(N(i)-1)/2,end);
    
    v100_ffd = v100((N(i)-1)/2,:,end);
    v400_ffd = v400((N(i)-1)/2,:,end);
    
    figure(h100)
    hold on
    plot(y_ffd, u100_ffd, 'linewidth', 2);
    
    figure(h400)
    hold on
    plot(y_ffd, u400_ffd, 'linewidth', 2);
    
end

figure(h100)
hold on
plot(y_paper,u100_paper, 'o', 'linewidth',2);

legend('FFD simulation (N=16)', 'FFD simulation (N=32)',...
    'FFD simulation (N=64)','Ghia, Ghia and Shin', 'location', 'NW');
title('u velocity at x = 0.5 for RE=100', 'fontweight',...
    'bold','fontsize', 15);
xlabel('y', 'fontsize', 12);
ylabel('u', 'fontsize', 12);
set(gcf,'units','centimeters','position',[0 0 20,20])

figure(h400)
hold on
plot(y_paper,u400_paper, 'o', 'linewidth', 2);

legend('FFD simulation (N=16)', 'FFD simulation (N=32)',...
    'FFD simulation (N=64)','Ghia, Ghia and Shin', 'location', 'NW');
title('u velocity at x = 0.5 for RE=400', 'fontweight',...
    'bold','fontsize', 15);
xlabel('y', 'fontsize', 12);
ylabel('u', 'fontsize', 12);
set(gcf,'units','centimeters','position',[0 0 20,20]);
end