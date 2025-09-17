%% This is a script to test the SVMD method
% Authors: Mojtaba Nazari and Sayed Mahmoud Sakhaei
% mojtaba.nazari.21@gmail.com -- smsakhaei@nit.ac.ir
% Initial release 2020-5-15 (c) 2020

close all
clear 
clc

%% Example 1 (ECG signal)
load('ECG.mat'); 
signal=(val (1,:));

%----------------- Initialization
maxAlpha=20000; %compactness of mode
tau=0;%time-step of the dual ascent
tol=1e-6; %tolerance of convergence criterion;
stopc=4;%the type of stopping criteria
fs=125; % sampling frequency 
T = length(signal);% time domain (t -->> 0 to T)
t = (1:T)/T;
omega_freqs = t-0.5-1/T;%discretization of spectral domain
f_hat=fftshift(fft(signal));

%-------------- SVMD function
[u,uhat]=svmd(signal,maxAlpha,tau,tol,stopc);

plot(signal)
hold on
plot(sum(u))
legend('Input Signal','Reconstructed Signal')
figure
subplot(211)
plot(omega_freqs*fs,abs(uhat))
title('Spectrum of reconstructed signal')
xlabel('Hz')
subplot(212)
plot(omega_freqs*fs,abs(f_hat))
title('Spectrum of original signal')
xlabel('Hz')
%% Example 2 (EEG signal)
clear all;
close all;

load('EEG.mat');
signal=double(eeg (1,:));


%% Initialization
maxAlpha=1000; %compactness of mode
tau=0;%time-step of the dual ascent
tol=1e-6; %tolerance of convergence criterion;
stopc=1;%the type of stopping criteria

fs=200; % sampling frequency 
T = length(signal);% time domain (t -->> 0 to T)
t = (1:T)/T;
omega_freqs = t-0.5-1/T;%discretization of spectral domain
f_hat=fftshift(fft(signal));

[u,uhat]=svmd(signal,maxAlpha,tau,tol,stopc);
% [u,~]= vmd(signal, 'NumIMFs',4,'PenaltyFactor',1000);
% u = u';


figure()
stackedplot(u')

figure()
plot(signal)
hold on
plot(sum(u))
legend('Input Signal','Reconstructed Signal')

figure
subplot(211)
plot(omega_freqs*fs,abs(uhat))
title('Spectrum of reconstructed signal')
xlabel('Hz')
subplot(212)
plot(omega_freqs*fs,abs(f_hat))
title('Spectrum of original signal')
xlabel('Hz')

