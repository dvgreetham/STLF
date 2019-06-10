setwd("Python\ scripts")
library(ggplot2)
data <- read.csv("myEPM.csv")
data <- data[-c(1),-c(7)]
head(data)

dat <- data[23331:26402,]
load <- dat[seq(7,232)]
n <- dim(dat)[1]
OBS <- load[(n-335):n,]
LW <- read.csv("LWforecast.csv",header=F)
SD <- read.csv("SD_4_forecast.csv",header=F)
AR <- read.csv("SARIMAorecast.csv",header=F)
MLP <- read.csv("MLP_forecast.csv",header=F)
LSTM <- read.csv("ANN_20-20-forecast-new.csv",header=F)

PM4 <- read.csv("PM_4_forecast.csv",header=F)

#scedasis
LW_abs <- abs(LW - OBS)
SD_abs <- abs(SD - OBS)
MLP_abs <-abs(MLP - OBS)
AR_abs <- abs(AR - OBS)
LSTM_abs <- abs(LSTM - OBS)
#PM3_abs <- abs(PM3 - OBS)
PM4_abs <- abs(PM4 - OBS)
#PM5_abs <- abs(PM5 - OBS)

lim_k <- 50
#########################       Biweight Kernel     ##########################
G_kernel <- function(u){
  if (abs(u)<=1){
    G <- 15*((1-u**2)**2)/16
  }else {
    G <- 0.0
  }
  return(G)
}

#########################       Last Week     ##########################
lw <- apply(LW_abs,1,max)
n <- length(lw)
data_ord_lw <- sort(lw) #create an ordered vector

ss <- seq(from=1/n,to=1,by=1/n)
c_lw <- c()

h <- 0.1 
k_lw <- lim_k
k_lw_largest <- sort(lw)[n-k_lw]

for (j in 1:floor(n*h)) {
  i <- 1:n
  x <- (j - i)/floor(n*h)
  s <- ss[j]
  p <- s/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- -(1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_lw[j] <- sum(g[lw>k_lw_largest])/(k_lw*h)
  print(j)
}

for (j in floor(n*h):(n-floor(n*h))) {
  i <- (1:n)
  u <- (j-i)/floor(n*h)
  c_lw[j] <- (sum((lw>k_lw_largest)*sapply(u,FUN=G_kernel)))/(k_lw*h)
}

for (j in (n-floor(n*h)):n) {
  s <- ss[j]
  x <- (j- 1:n)/floor(n*h)
  p <- (1-s)/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- (1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_lw[j] <- sum(g[lw>k_lw_largest])/(k_lw*h)
}


#########################       Similar Day     ##########################
sd <- apply(SD_abs,1,max)
n <- length(sd)
data_ord_sd <- sort(sd) #create an ordered vector

ss <- seq(from=1/n,to=1,by=1/n)
c_sd <- c()

h <- 0.1 
k_sd <- lim_k
k_sd_largest <- sort(sd)[n-k_sd]

for (j in 1:floor(n*h)) {
  i <- 1:n
  x <- (j - i)/floor(n*h)
  s <- ss[j]
  p <- s/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- -(1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_sd[j] <- sum(g[sd>k_sd_largest])/(k_sd*h)
  print(j)
}

for (j in floor(n*h):(n-floor(n*h))) {
  i <- (1:n)
  u <- (j-i)/floor(n*h)
  c_sd[j] <- (sum((sd>k_sd_largest)*sapply(u,FUN=G_kernel)))/(k_sd*h)
}

for (j in (n-floor(n*h)):n) {
  s <- ss[j]
  x <- (j- 1:n)/floor(n*h)
  p <- (1-s)/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- (1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_sd[j] <- sum(g[sd>k_sd_largest])/(k_sd*h)
}


#########################       MLP forecast     ##########################
mlp <- apply(MLP_abs,1,max)
n <- length(mlp)
data_ord_mlp <- sort(mlp) #create an ordered vector

ss <- seq(from=1/n,to=1,by=1/n)
c_mlp <- c()

h <- 0.1 
k_mlp <- lim_k
k_mlp_largest <- data_ord_mlp[n-k_mlp]

for (j in 1:floor(n*h)) {
  i <- 1:n
  x <- (j - i)/floor(n*h)
  s <- ss[j]
  p <- s/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- -(1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_mlp[j] <- sum(g[mlp>k_mlp_largest])/(k_mlp*h)
}

for (j in floor(n*h):(n-floor(n*h))) {
  i <- (1:n)
  u <- (j-i)/floor(n*h)
  c_mlp[j] <- (sum((mlp>k_mlp_largest)*sapply(u,FUN=G_kernel)))/(k_mlp*h)
}

for (j in (n-floor(n*h)):n) {
  s <- ss[j]
  x <- (j- 1:n)/floor(n*h)
  p <- (1-s)/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- (1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_mlp[j] <- sum(g[mlp>k_mlp_largest])/(k_mlp*h)
}

 

#########################       PM4 forecast     ##########################
pm4 <- apply(PM4_abs,1,max)
n <- length(pm4)
data_ord_pm4 <- sort(pm4) #create an ordered vector

ss <- seq(from=1/n,to=1,by=1/n)
c_pm4 <- c()

h <- 0.1 
k_pm4 <- lim_k
k_pm4_largest <- data_ord_pm4[n-k_pm4]

for (j in 1:floor(n*h)) {
  i <- 1:n
  x <- (j - i)/floor(n*h)
  s <- ss[j]
  p <- s/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- -(1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_pm4[j] <- sum(g[pm4>k_pm4_largest])/(k_pm4*h)
}

for (j in floor(n*h):(n-floor(n*h))) {
  i <- (1:n)
  u <- (j-i)/floor(n*h)
  c_pm4[j] <- (sum((pm4>k_pm4_largest)*sapply(u,FUN=G_kernel)))/(k_pm4*h)
}

for (j in (n-floor(n*h)):n) {
  s <- ss[j]
  x <- (j- 1:n)/floor(n*h)
  p <- (1-s)/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- (1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_pm4[j] <- sum(g[pm4>k_pm4_largest])/(k_pm4*h)
}


#########################       SARIMA forecast     ##########################

ar <- apply(AR_abs,1,max)
n <- length(ar)
data_ord_ar <- sort(ar) #create an ordered vector

ss <- seq(from=1/n,to=1,by=1/n)
c_ar <- c()

h <- 0.1 
k_ar <- lim_k
k_ar_largest <- data_ord_ar[n-k_ar]

for (j in 1:floor(n*h)) {
  i <- 1:n
  x <- (j - i)/floor(n*h)
  s <- ss[j]
  p <- s/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- -(1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_ar[j] <- sum(g[ar>k_ar_largest])/(k_ar*h)
}

for (j in floor(n*h):(n-floor(n*h))) {
  i <- (1:n)
  u <- (j-i)/floor(n*h)
  c_ar[j] <- (sum((ar>k_ar_largest)*sapply(u,FUN=G_kernel)))/(k_ar*h)
}

for (j in (n-floor(n*h)):n) {
  s <- ss[j]
  x <- (j- 1:n)/floor(n*h)
  p <- (1-s)/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- (1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_ar[j] <- sum(g[ar>k_ar_largest])/(k_ar*h)
}



#########################       LSTM forecast     ##########################
lstm <- apply(LSTM_abs,1,max)
n <- length(lstm)
data_ord_lstm <- sort(lstm) #create an ordered vector

ss <- seq(from=1/n,to=1,by=1/n)
c_lstm <- c()

h <- 0.1 
k_lstm <- lim_k
k_lstm_largest <- data_ord_lstm[n-k_lstm]

for (j in 1:floor(n*h)) {
  i <- 1:n
  x <- (j - i)/floor(n*h)
  s <- ss[j]
  p <- s/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- -(1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_lstm[j] <- sum(g[lstm>k_lstm_largest])/(k_lstm*h)
}

for (j in floor(n*h):(n-floor(n*h))) {
  i <- (1:n)
  u <- (j-i)/floor(n*h)
  c_lstm[j] <- (sum((lstm>k_lstm_largest)*sapply(u,FUN=G_kernel)))/(k_lstm*h)
}

for (j in (n-floor(n*h)):n) {
  s <- ss[j]
  x <- (j- 1:n)/floor(n*h)
  p <- (1-s)/h
  a0 <- (1/16)*((p+1)^3)*(3*p^2 - 9*p + 8)
  a1 <- (1/16)*(5/2)*(p^2 -1)^3
  a2 <- (1/16)*(1/7)*(15*p^7 - 42*p^5 + 35*p^3 + 8)
  g <- ((a2 - x*a1)/(a0*a2 - a1^2))*sapply(x,G_kernel)
  c_lstm[j] <- sum(g[lstm>k_lstm_largest])/(k_lstm*h)
}

sced <-data.frame(lw=c_lw,sd=c_sd, ar=c_ar,lstm=c_lstm,pm1=c_pm4, mlp=c_mlp)
write.csv(sced, 'sced50.csv')

ggplot(sced)  + 
  geom_line(aes(x=1:336,y=lw,col="LW"))+ 
   geom_line(aes(x=1:336,y=sd,col="SD4")) +
  geom_line(aes(x=1:336,y=ar,col="SARIMA")) + 
  geom_line(aes(x=1:336,y=lstm,col="LSTM")) +
 
  geom_line(aes(x=1:336,y=pm1,col="PM4")) +
#   geom_line(aes(x=1:336,y=pm2,col="PM5")) +
  geom_line(aes(x=1:336,y=mlp,col="MLP")) +
  xlab("") +ylab(expression(hat(c))) + ylim(c(0,2.5))+
  geom_hline(yintercept = 1) + theme(legend.title=element_blank())+
  theme(axis.title=element_text(size=13), axis.text = element_text(size=11)) +
  scale_x_continuous(breaks = seq(12,336,by=48),
                     labels=c("Mon","Tue","Wed",
                              "Thur","Fri","Sat","Sun"))
