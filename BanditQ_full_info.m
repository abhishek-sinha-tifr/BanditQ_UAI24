% This code simulates the BanditQ policy in the full information setting
clear all;
rng(100); % setting seed for random number generation
T=3*10^6; % Horizon length
N=1000; % number of arms

%mu=[0.3352    0.2031    0.2411    0.7816    0.6177];
%mu=[0.3835    0.5031    0.2411    0.7816    0.6177];
%mu=[0.05, 0.5, 0.3, 0.8, 0.1];

mu=rand(1,N); 
%load("mu.mat");

lambda=zeros(N,1);

k=2; % number of arms with reward constraints
% Setting the target rates
lambda(1)=mu(1)/2; 
lambda(2)=mu(2)/3;
%lambda(2)=mu(2)/3;

% Computing the optimal fraction of pulls in retrospect

[v, best_arm]=max(mu);
opt_frac=lambda'./mu;

if(opt_frac(best_arm)==0)
  opt_frac(best_arm)= 1-sum(opt_frac);
end


% Defining the queue variables
Q=zeros(1,N);
r_avg=zeros(1,N);

cum_reward=zeros(1,T+1); % tracking the cumulative reward accrued by BanditQ

q_log=zeros(k,T); % logging sum of queue lengths
r_log=zeros(k,T);

% simulating the full-information case 
S=0; % the running sum of square of the gradients 
x=(1/N)*ones(1,N); % initialiazing the sampling distribution

progressbar;

for t=1:T
    progressbar(t/T); % showing the ETA
    r=2*mu.*rand(1,N); % generating the reward vector with mean mu. Note that the rewards lie in the interval [0,2]
    V=sqrt(T);
    r_avg=r_avg+(1/t)*(x.*r-r_avg);
    cum_reward(t+1) = cum_reward(t) + r*x'; % updating the cumulative rewards
    

    for i=1:k %updating the queue-lengths
        Q(i)=max(0,Q(i)+lambda(i)-r(i)*x(i));
    end

    g=r.*(Q+V); % computing the gradients
    S=S+norm(g)^2;
    x=projsplx(x+g/sqrt(2*S)); % computing the prediction distribution for the subsequent round
   % x=x'; % converting it to a row vector 
    
    for i=1:k
     q_log(i,t)=Q(i); 
     r_log(i,t)=r_avg(i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 figure(1)
 plot(1:T, q_log(1,:),'color', [0 0.4470 0.7410], 'LineWidth', 2.5, 'DisplayName', ' Arm 1');
 hold on;
 plot(1:T, q_log(2,:),'color', [0.8500 0.3250 0.0980], 'LineWidth', 2.5, 'DisplayName', ' Arm 2');
 %hold on;
 %plot(1:T, q_log(1,:)+q_log(2,:),'color', [0.4940 0.1840 0.5560], 'LineWidth', 1.5, 'DisplayName', ' Sum of queue lengths');
 xlabel('Number of Rounds', 'FontSize', 18);
 ylabel('Queue lengths', 'FontSize', 18);
 %title('Evolution of Queue lengths (Bandit Feedback)', 'FontName','Helvetica', 'FontSize', 16);
 lgd=legend('Location', 'best', 'FontSize', 18);
 set(lgd,'Box','off');
 set(gca,'FontSize',18)
 save2pdf('Q_lengths_full_info.pdf');


 figure(2)
 %title('Reward rates for Arm 1 and 2 in the Bandit feedback setting');
 plot(1:T, r_log(1,:), 'color', [0 0.4470 0.7410], 'LineWidth', 3.5, 'DisplayName', ' Arm 1 rewards');
 hold on;
 plot(1:T, lambda(1)*ones(1,T), ':', 'color', [0 0.4470 0.7410], 'LineWidth', 3.5, 'DisplayName', ' Arm 1 target rate')
 hold on;
 plot(1:T, r_log(2,:), '-.', 'color', [0.8500 0.3250 0.0980], 'LineWidth', 3.5, 'DisplayName', ' Arm 2 rewards');
 hold on;
 plot(1:T, lambda(2)*ones(1,T), '--', 'color', [0.8500 0.3250 0.0980], 'LineWidth', 3.5, 'DisplayName', ' Arm 2 target rate');
 axis([0 T 0 0.5])
 xlabel('Number of Rounds', 'FontSize', 18);
 ylabel('Reward accrual rate', 'FontSize', 18);
 lgd=legend('Location', 'best', 'FontSize', 18);
 set(lgd,'Box','off');
 set(gca,'FontSize',18)

 save2pdf('Reward_rates_full_info.pdf');

 figure(3)
 %title('Regret incurred by the BanditQ policy');
 %plot(1:T, S4(2:T+1,:)*opt_frac'-S3(2:T+1)');
 plot(1:T,opt_frac*mu'*[1:T]-cum_reward(2:T+1), 'color', [0.4940 0.1840 0.5560]	, 'LineWidth', 3, 'DisplayName', ' Regret');
 xlabel('Number of Rounds', 'FontSize', 18);
 ylabel('Regret', 'FontSize', 18);
 grid;
 set(gca,'FontSize',18)

 save2pdf('Regret_full_info.pdf');







