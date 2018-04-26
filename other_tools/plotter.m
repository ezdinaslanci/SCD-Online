cs = load('change_scores.txt');
is = load('input_sequence.txt');

figure;

ah(1) = subplot(2,1,1);
plot(is); 
grid on;
set(gca,'xtick', [1479]);
xlabel("Time");
ylabel("Token");
title("Pre-processed Demonstrator Data with 2 Different Colored Lego Pieces");

ah(2) = subplot(2,1,2);
plot(cs); 
grid on;
set(gca,'xtick', [1479]);
xlabel("Time");
ylabel("Change Score");
title("Predicted Change Scores of SCD");

linkaxes(ah, 'x');