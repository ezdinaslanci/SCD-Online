function [] = plotMPs (mps, alpha, order)

    z = 1 : alpha;
    f = alpha + 1 : alpha + alpha^2;
    s = alpha + alpha^2 + 1 : alpha + 2 * alpha^2;
    
    figure;
    
    if (order == 0)
        for i = 1 : alpha
            dn = ['P(' num2str(i) ')'];
            plot(mps(:, i), 'DisplayName', dn);
            hold on;
        end
    end
    
    if (order == 1)
        for i = 1 : alpha
            for j = 1 : alpha
                dn = ['P(' num2str(i) '|' num2str(j) ')'];
                plot(mps(:, i * alpha + j), 'DisplayName', dn);
                hold on;
            end
        end
    end
    
    
    legend('show');
    grid on;

end