%  Traning Feed-forward Neural Networks using LCGSA   %
%                                                                   %
%  Developed in MATLAB R2013b                                       %
%                                                                   %
%  programmer: Sajad Ahmad Rather                        %
%                                                                   %
%         e-Mail: sajad.win8@gmail.com                              %
%                                                                   %
% Homepage: https://www.linkedin.com/in/sajad-ahmad-rather-97a398110/   %
%  

% clc
% close all
% clear all

function [CFbest,CLbest,CBestChart]= LCGSA(n, iteration, min_flag,low,up,dim,fobj,chaosIndex) 
% %%%%
%  

G0=1;                                          % gravitational constant
current_position = rand(n,dim).*(up-low)+low;  %initial positions in the problem's boundary
velocity = .3*randn(n,dim) ;
acceleration=zeros(n,dim);
mass(n)=0;
force=zeros(n,dim);

% if size(up,1)==1
% for i = 1:dim
%     lambda = rand(n,1);
%     current_position(:,i) = fix(low(i,1) + lambda * (up(i,1) - low(i,1)));
%       current_position= sort(current_position);
% end

% tic
chValueInitial=20;
wMax=chValueInitial;
wMin=1e-10;
iter = 0 ;  
CBestChart=[];% Iteration counter

while  ( iter < iteration )
    
iter = iter + 1;
iter;
 chValue=wMax-iter*((wMax-wMin)/iteration);
 G=G0*exp(-23*iter/iteration); %Equation (4)

switch chaosIndex
  
    case 1
          G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 2
         G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 3
        G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 4
        G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 5
        G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 6
        G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 7
        G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 8
        G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 9
        G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
    case 10
        G=G+Levy1(n,1)+chaos(chaosIndex,iter,iteration,chValue);
  otherwise
        warning('Unexpected plot type. No plot created.')                
end
 
%  velocity = .3*randn(n,dim) ;
%  force=zeros(n,dim);
%  mass(n)=0;
%  acceleration=zeros(n,dim);
%    [dim,n]=size(current_position);

 for i = 1:n
%      [n,dim]=size(current_position);
%     %///Bound the search Space///
fitness =0;
Tp=current_position(i,:)>up;Tm=current_position(i,:)<low;current_position(i,:)=(current_position(i,:).*(~(Tp+Tm)))+up.*Tp+low.*Tm; 
%     Tp=current_position>up;
%     
%     Tm=current_position<low;
%     current_position=(current_position .*(~(Tp+Tm)))+up.*Tp+low.*Tm; 
%     %%
%     [N,dim]=size(X);
% for i=1:n 
% Tp=X(i,:)>up;Tm=X(i,:)<low;X(i,:)=(X(i,:).*(~(Tp+Tm)))+up.*Tp+low.*Tm;
%     %%Agents that go out of the search space, are reinitialized randomly .
%     Tp=current_position(i,:)>up;Tm=current_position(i,:)<low;current_position(i,:)=(current_position(i,:).*(~(Tp+Tm)))+((rand(1,dim).*(up-low)+low).*(Tp+Tm));
    %%
    %////////////////////////////
    
                                 %-------------------------------------------%
                                 %         Evaluate the population           %           
                                 %-------------------------------------------% 
%        fitness=0;
%     (m,level,xR,PI)
    fitness=fobj(current_position(i,:));
    current_fitness(i)=fitness;   
        
%     if(pBestScore(i)>fitness)
%         pBestScore=fitness;
%         pBest(i,:)=current_fitness(:,i);
%     end
%     if(gBestScore>fitness)
%         gBestScore=fitness;
%         gBest=current_position(:,i);
%     end
%        current_position= sort(current_position);
    %evalua x en la funcion objetivo
    %[fbest, fitBestR] = fitnessIMG(I, 1, Lmax, level, x, probR);
%      current_fitness = Kapur(1,level,current_position,probR);
 end

%  best=max(current_fitness);
%  worst=min(current_fitness);

%         GlobalBestCost=gBestScore;
%         GlobalBestCost;
%         best;
%         
%         
% 
%     for pp=1:n
%         if current_fitness(pp)==best
%             break;
%         end
%         
%     end
%     
%     bestIndex=pp;
%             
%     for pp=1:dim
%         best_fit_position(iter,1)=best;
%         best_fit_position(iter,pp+1)=current_position(bestIndex,pp);   
%     end
%  if min_flag==1
    [best, best_current_position]=min(current_fitness); %minimization.
%      else
    [worst, best_current_position]=max(current_fitness); %maximization.
%  end        
    
if iter==1
       CFbest=best;
       CLbest=current_position(best_current_position,:);
end
if min_flag==1
      if best<CFbest  %minimization.
       CFbest=best;
       CLbest=current_position(best_current_position,:);
       end
     else 
       if best>CFbest  %maximization
        CFbest=best;
        CLbest=current_position(best_current_position,:);
       end
end
      
CBestChart=[CBestChart CFbest];

%  disp(['Iteration ' num2str(iter) ': GlobalBestCost = ' num2str(GlobalBestCost)]);
                                               %-------------------%
                                               %   Calculate Mass  %
                                               %-------------------%
    for i=1:n
    mass(i)=(current_fitness(i)-0.99*worst)/(best-worst);    
end

for i=1:n
    mass(i)=mass(i)*5/sum(mass);    
    
end

                                               %-------------------%
                                               %  Force    update  %
                                               %-------------------%

for i=1:n
    for j=1:dim
        for k=1:n
            if(current_position(k,j)~=current_position(i,j))
                % Equation (3)
                 force(i,j)=force(i,j)+ rand()*G*mass(k)*mass(i)*(current_position(k,j)-current_position(i,j))/abs(current_position(k,j)-current_position(i,j));
                
            end
        end
    end
end
                                               %------------------------------------%
                                               %  Accelations $ Velocities  UPDATE  %
                                               %------------------------------------%

for i=1:n
       for j=1:dim
            if(mass(i)~=0)
                %Equation (6)
                acceleration(i,j)=force(i,j)/mass(i);
            end
       end
end   


for i=1:n
        for j=1:dim
            %Equation(9)
            velocity(i,j)=rand()*velocity(i,j)+rand()*acceleration(i,j);
%             size(velocity)
         % Apply Velocity Limits
%        velocity(j,i) = max(velocity(j,i),VelMin);
%        velocity(j,i) = min(velocity(j,i),VelMax);
        end

      
end
                                               %--------------------------%
                                               %   positions   UPDATE     %
                                               %--------------------------%
%  for i=1:n
%         for j=1:dim                                                       
%Equation (10) 
current_position = current_position + velocity ;

   
   
end
%   toc

%  plot(GS_Fit_bests)
end







