% crane forward and inverse kinematic
function crane_kinematic()
    clear all; close all;
    syms theta1 theta2

    % initial configuration
    L_base = 1.2;
    W_base = 0.5;
    L_boom_a = 2.0;
    W_boom_a = 0.1;
    L_boom_b = 1.0;
    W_boom_b = 0.08;

    theta1_0=  0.0*pi;
    theta2_0 = -0.7*pi;
    theta_step = 5 * pi/180;
    x_step = 0.05;

    Lhyd1 = 0.7810;
    Lhyd2 = 0.6510;

    Tworld = eye(4);
    Tab_sym = [rotz(theta1) [0;L_base;0]; 0 0  0 1];
    Tbc_sym = [rotz(theta2) [L_boom_a;0;0]; 0 0  0 1];
    Tce = [rotz(0) [L_boom_b;0;0]; 0 0  0 1];

    Tj1 =[eye(3) [0.3;0.5*L_base;0]; 0 0  0 1];
    Tj2_sym =Tab_sym*[rotz(theta1) [0.4*L_boom_a;0;0]; 0 0  0 1];
    Tj3_sym =Tab_sym*[rotz(theta1) [0.6*L_boom_a;0;0];0 0  0 1];
    Tj4_sym =Tab_sym*Tbc_sym*[rotz(theta2) [0.4*L_boom_b;0;0]; 0 0  0 1];

    THETA = [theta1_0,theta2_0];
    xy = compute_forward(Tworld,Tab_sym,Tbc_sym,Tce,theta1_0,theta2_0);
    X = [xy];

    theta1 = THETA(end,1);
    theta2 = THETA(end,2);
    hold on
    plot_crane(Tworld, eval(Tab_sym), eval(Tbc_sym), Tce, Tj1, eval(Tj2_sym), eval(Tj3_sym), eval(Tj4_sym),Lhyd1, Lhyd2, W_base, L_base, W_boom_a, L_boom_a, W_boom_b, L_boom_b)
    xlim([-0.6,3.5])
    ylim([-0.1,4])
    axis equal
    hold off

    while 1
        w = waitforbuttonpress;
        [q_add,x_add,bool_quit,bool_q_add] = evaluate_key_press(w,x_step,theta_step);
        if bool_quit
            close all;
            return
        end
        
        if bool_q_add
            THETA = [THETA;...
                     THETA(end,:) + q_add];
            theta1 = THETA(end,1);
            theta2 = THETA(end,2);
            xy = compute_forward(Tworld,Tab_sym,Tbc_sym,Tce,theta1,theta2);
            X = [X;xy];
        else
            theta1_0=  THETA(end,1);
            theta2_0 = THETA(end,2);
            x_dest = (X(end,:)+x_add)';
            q = compute_inverse(Tworld,Tab_sym,Tbc_sym,Tce,theta1_0,theta2_0,x_dest);
            X = [X;compute_forward(Tworld,Tab_sym,Tbc_sym,Tce,q(1),q(2));];
            theta1 = q(1);
            theta2 = q(2);
            THETA = [THETA; theta1,theta2];
        end
        
        clf;
        hold on
        plot(X(:,1),X(:,2),'-b','linewidth',2)
        plot_crane(Tworld, eval(Tab_sym), eval(Tbc_sym), Tce, Tj1, eval(Tj2_sym), eval(Tj3_sym), eval(Tj4_sym),Lhyd1, Lhyd2, W_base, L_base, W_boom_a, L_boom_a, W_boom_b, L_boom_b)
        xlim([-0.6,3.5])
        ylim([-0.1,4])
        axis equal
        hold off
    end
end

function [q_new] = compute_inverse(Tworld,Tab_sym,Tbc_sym,Tce,theta1_0,theta2_0,x_dest)
    syms theta1 theta2
    
    q = [theta1;theta2];
    Twe = Tworld * Tab_sym * Tbc_sym * Tce;
    X_sym = [Twe(1,4); Twe(2,4)];
 
    J_sym = jacobian(X_sym,q);

    lambda = 0.9;
    err_tol = 0.001;
    m_iter = 0;
    max_iter = 100;
    
    theta1 = theta1_0;
    theta2 = theta2_0;

    err = x_dest - eval(X_sym);
    q_old = eval(q);

    while (norm(err)>=err_tol) && (m_iter<max_iter)
        J = eval(J_sym);
        delta_theta = pinv(J)*(lambda*err);
        q_new = q_old + delta_theta;

        theta1 = q_new(1);    
        theta2 = q_new(2);
        
        q_old = q_new;
        err = x_dest - eval(X_sym);
        m_iter = m_iter + 1;
    end
    disp(['stopped after iter = ',num2str(m_iter)])
    
end

function [xy] = compute_forward(Tworld,Tab_sym,Tbc_sym,Tce,theta1_num,theta2_num)
    theta1 = theta1_num;
    theta2 = theta2_num;
    Twe = Tworld * eval(Tab_sym) * eval(Tbc_sym) * Tce;
    xy = [Twe(1,4), Twe(2,4)];
end

function [q_add,x_add,bool_quit,bool_q_add] = evaluate_key_press(w,x_step,theta_step)
    if w
           p = get(gcf, 'CurrentCharacter');
    end
    bool_q_add = 0;
    bool_quit = 0;
    
    q_add = [0,0];
    x_add = [0,0];
    
    switch double(p)
        % theta changes
        case 102
            disp('f: theta 1 down');
            q_add(1) = -theta_step;
            bool_q_add = 1;
        case 114
            disp('r: theta 1 up');
            q_add(1) = theta_step;
            bool_q_add = 1;
        case 103
            disp('g: theta 2 down');
            q_add(2) = -theta_step;
            bool_q_add = 1;
        case 116
            disp('t: theta 2 up');
            q_add(2) = theta_step;
            bool_q_add = 1;
            
        % x changes
        case 97
            disp('a: x left');
            x_add(1) = -x_step;
        case 100
            disp('d: x right');
            x_add(1) = x_step;
        case 115
            disp('s: y down');
            x_add(2) = -x_step;
        case 119
            disp('w: y up');
            x_add(2) = x_step;
        case 113
            disp('q quit');
            bool_quit = 1;
    end
end

function plot_crane(Tworld,Tab,Tbc,Tce,Tj1,Tj2,Tj3,Tj4,Lhyd1,Lhyd2,W_base,L_base,W_boom_a,L_boom_a,W_boom_b,L_boom_b)

    X_base = [-W_base, W_base, W_base,0.1,-0.1,-W_base;...
               0, 0, 0.1, L_base+0.2, L_base+0.2, 0.1];
    X_base = [X_base;zeros(1,size(X_base,2))];

    X_boom_A = [-W_boom_a -W_boom_a;...
                W_boom_a+L_boom_a -W_boom_a;...
                W_boom_a+L_boom_a W_boom_a;...
                -W_boom_a W_boom_a]';
    X_boom_A = [X_boom_A;zeros(1,size(X_boom_A,2))];

    X_boom_B = [-W_boom_b -W_boom_b;...
                W_boom_b+L_boom_b -W_boom_b;...
                W_boom_b+L_boom_b W_boom_b;...
                -W_boom_b W_boom_b]';
    X_boom_B = [X_boom_B;zeros(1,size(X_boom_B,2))];

    plot_patch(X_base,Tworld,'r')
    plot_patch(X_boom_A,Tworld*Tab,'m')
    plot_patch(X_boom_B,Tworld*Tab*Tbc,'c')

    plot_joint(Tj1,0.05)
    plot_joint(Tj2,0.05)
    plot_joint(Tj3,0.05)
    plot_joint(Tj4,0.05)

    plot_joint(Tworld*Tab,0.05)
    plot_joint(Tworld*Tab*Tbc,0.05)
    plot_joint(Tworld*Tab*Tbc*Tce,0.05)

    plot_hydraulic_piston_2D(Tj1,Tj2,Lhyd1,0.12*Lhyd1,'k',2)
    plot_hydraulic_piston_2D(Tj3,Tj4,Lhyd2,0.15*Lhyd2,'k',2)
end

function plot_patch(X,T,color)
    X1 = T*[X;ones(1,size(X,2))];
    patch(X1(1,:),X1(2,:),color);
end

function plot_joint(T,rad)
    n = 25;
    Xa = (T*[0,rad*cos(linspace(0,0.5*pi,n));0,rad*sin(linspace(0,0.5*pi,n));zeros(1,n+1);ones(1,n+1)])';
    Xb = (T*[0,rad*cos(linspace(1*pi,1.5*pi,n));0,rad*sin(linspace(1*pi,1.5*pi,n));zeros(1,n+1);ones(1,n+1)])';
    Xc = (T*[rad*cos(linspace(0,2*pi,n));rad*sin(linspace(0,2*pi,n));zeros(1,n);ones(1,n)])';
    patch(Xc(:,1),Xc(:,2),'k')
    patch(Xa(:,1),Xa(:,2),'w')
    patch(Xb(:,1),Xb(:,2),'w')
end

function plot_hydraulic_piston_2D(T1,T2,L0,d,color,lw)
    x1 = T1(1:2,4)';    
    x2 = T2(1:2,4)';
    X = [x1;x2];
    mag = sqrt(sum(diff(X).^2));    
    vdir =(X(2,:)-X(1,:))./mag;
    vnorm = [vdir(2),-vdir(1)];
    x3 = x1+0.1*L0*vdir;
    x4 = x3+0.5*d*vnorm;    
    x5 = x3-0.5*d*vnorm;
    x6 = x4+0.8*L0*vdir;    
    x7 = x5+0.8*L0*vdir;
    x8 = x2-0.88*L0*vdir;
    x9 = x8+0.3*d*vnorm;    
    x10 = x8-0.3*d*vnorm;
    px = [x1(1), x3(1), x3(1), x4(1), x5(1), x2(1), x8(1), x8(1) ;...
          x3(1), x4(1), x5(1), x6(1), x7(1), x8(1), x9(1), x10(1) ];
    py = [x1(2), x3(2), x3(2), x4(2), x5(2), x2(2), x8(2), x8(2);...
          x3(2), x4(2), x5(2), x6(2), x7(2), x8(2), x9(2), x10(2)];
    plot(px,py,color,'linewidth',lw)
end

function [R] =rotz(theta)
    R = [cos(theta) -sin(theta) 0;...
         sin(theta)  cos(theta) 0;...
            0           0       1];
end
