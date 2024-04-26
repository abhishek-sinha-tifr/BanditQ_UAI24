% This code simulates the BanditQ policy in the bandit feedback setting
clear all;
rng(100); %setting seed for random number generation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializations and defining bookkeeping variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


T=3*10^6; % Horizon length
N=1000; % number of arms
%mu=[0.3352    0.2031    0.2411    0.7816    0.6177];
%mu=[0.05, 0.5, 0.3, 0.8, 0.1];mu=rand(1,N);
%load("mu.mat");
mu=rand(1,N);
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

eta=N;
gamma = 0.5; % initializing the parameters used in the Bandit algorithm


q_log=zeros(k,T); % logging sum of queue lengths
r_log=zeros(k,T); % logging the instantaneous rates
% simulating the full-information case 
S2=zeros(1,N); % running sum of surrogate rewards 
S1=1; % auxiliary sum used for setting the eta parameter
p=(1/N)*ones(1,N); % initialiazing the sampling distribution
NumSample=zeros(1,N); % initializing the number of times an arm was sampled 
% Arrays used for computing the regret
S3=zeros(1,T+1); % array to track the cumulative reward accrued by banditQ
%S4=zeros(T+1,N); % array to track the cumulative rewards of the arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  The Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code for displaying the simulation progress
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
progressbar;
for t=1:T
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    progressbar(t/T);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The main loop starts below
    %%%%%%%%%%%%%%%%%%%%   
    X=zeros(1,N); % denoting the selected user
    %p_dash=zeros(1,N);
    %if(sum(Q)==0)
    %    X(best_arm)=1;
    %else
    p_dash=(1-gamma)*p + gamma/N; % defining the sampling distribution with exploration
    gamma = min(0.5, sqrt(N/t)); % setting the gamma parameter
    [val, i_t]=max(cumsum(p_dash)>rand()); % sampling an arm according to the distribution p
    X(i_t)=1;
    
    %i_t=randsample(1:N,1,true, p_dash);
    r=2*mu.*rand(1,N); % generating the reward vector with mean mu. Note that the rewards lie in the interval [0,2]
    %V=sqrt(T);
    V=sqrt(T);
  
  
    for i=1:k %updating the queue-lengths
        Q(i)=max(0,Q(i)+lambda(i)-r(i)*X(i)); 
    end

    
    
    r_avg=r_avg+(1/t)*(X.*r-r_avg); % updating the average rewards 
    r_dash=X.*r.*(Q+V)./p_dash; % computing the surrogate reward vector estimates fed to the bandit policy 
    S2 = S2+ r_dash; % accumulating the surrogate reward vector

    % Variables used only for reporting the regret
    [z2,i_t]=max(X);
    S3(t+1)=S3(t)+ r(i_t); % updating the cumulative reward accrued by BanditQ 
    %S4(t+1,:)=S4(t,:) + r; % updating the optimal offline cumulative reward 
    %%%%%%%%%%%%%%%%%%%%%%%
    % Implementing the FTRL step
    %%%%%%%%%%%%%%%%%%%%%%%
    z=eta*r_dash-1./p;
    [opt_val,opt_pt]=log_opt(z);
    opt_val= (1/eta)*(opt_val + N - eta*p*r_dash' - sum(log(p))); % call to the optimization module
    S1=S1+opt_val;
    eta = N/S1; % updating the parameter eta
    [temp,p]=log_opt(eta*S2); % assigning the new probability p by calling the optimization module again

 % logging various quantities of interest
    for i=1:k
     q_log(i,t)=Q(i); 
     r_log(i,t)=r_avg(i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% End of the main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%save('banditQ_bandit_vars');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %t= datetime;
 %mkdir(string(t));
 
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
 save2pdf('Q_lengths_Bandit_feedback.pdf');
 %z=fullfile(strcat(pwd, string(t)), 'Q_lengths_Bandit_feedback.pdf');
 %movefile('Q_lengths_Bandit_feedback.pdf',z);

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
 save2pdf('Reward_rates_bandit_feedback.pdf');

 figure(3)
 title('Regret incurred by the BanditQ policy');
 %plot(1:T, S4(2:T+1,:)*opt_frac'-S3(2:T+1)');
 plot(1:T,opt_frac*mu'*[1:T]-S3(2:T+1), 'color', [0.4940 0.1840 0.5560]	, 'LineWidth', 3, 'DisplayName', ' Regret');
 xlabel('Number of Rounds', 'FontSize', 18);
 ylabel('Regret', 'FontSize', 18);
 grid;
 set(gca,'FontSize',18)
 save2pdf('Regret_bandit_feedback.pdf');
 
%  % Comparing the reward accrued by the BanditQ and the LFG and Heuristic
%  % LFG policies 
%  T=3*10^5;
%  figure(4)
%  load('LFG_reward.mat');
%  load('LFG_reward_heuristic.mat');
%  plot(1:T,S3(2:T+1),'color', [0 0.4470 0.7410], 'LineWidth', 2.5, 'DisplayName', ' BanditQ rewards');
%  hold on;
%  plot(1:T, LFG_reward(2:T+1),'color', [0.8500 0.3250 0.0980], 'LineWidth', 2.5, 'DisplayName', ' LFG rewards');
% % hold on;
% % plot(1:T, LFG_reward_heuristic(2:T+1),'color', [0.4940 0.1840 0.5560]	, 'LineWidth', 2.5, 'DisplayName', ' LFG rewards');
%  xlabel('Number of Rounds', 'FontSize', 18);
%  ylabel('Cumuative Rewards', 'FontSize', 18);
%  lgd=legend('Location', 'best', 'FontSize', 18);
%  grid;
%  set(lgd,'Box','off');
%  set(gca,'FontSize',18);
%  save2pdf('Reward_comparison.pdf');



