################################################################################################
# LIBRARIES

library(tidyverse)
library(tsibble)
library(feasts)
library(TSA)
library(Hmisc)
library(astsa)
library(dynlm)
library(tsoutliers)
library(xts)
library(forecast)
library(urca)
library(tseries)

################################################################################################
# READ DATA

#Leemos nuestra data y creamos un df con las dos series juntas. Función ts para crear una serie temporal. Dividimos nuestras dos series temporales:
  
df <- read_csv("../practica_2/colgate_crest.csv")
colgate <- ts(df$Colgate, start = c(1958, 1), frequency = 52.18)
crest <- ts(df$Crest, start = c(1958, 1), frequency = 52.18)
data <- ts(cbind(colgate, crest))

################################################################################################
# PLOTS

#Dibujamos nuestras series temporales. Usamos autoplot, que es como ggplot con series temporales, sin necesidad de decirle que se trata de una serie temporal.
#Vemos que existe TENDENCIA en ambas series y en sentidos contrarios, siendo crest de tendencia creciente y colgate de tendencia decreciente.

autoplot(as_tsibble(colgate))
autoplot(as_tsibble(crest))

autoplot(data)+
  labs(title = "Series temporales Colgate y Crest",
       x="Semanas")

################################################################################################
# TRAIN/TEST

#A continuación dividimos nuestra muestra con train y test: Para ello, en el train nos quedamos con todas las semanas menos las 16 últimas, y en el tes 
#con las 16 últimas:
  
ts_colgate_train <- head(colgate, length(colgate) - 16)
ts_crest_train <- head(crest, length(crest) - 16)

ts_colgate_test <- tail(colgate, 16)
ts_crest_test <- tail(crest, 16)

ts_data_train <- cbind(ts_colgate_train, ts_crest_train)
ts_data_test <- cbind(ts_colgate_test, ts_crest_test)

autoplot(ts_colgate_train) + autolayer(ts_colgate_test)
autoplot(ts_crest_train) + autolayer(ts_crest_test)

################################################################################################
# AUTOARIMA COLGATE + PRED 16 SEMANAS

#The key of an autoarima model is we are trying to explain future values based on past values of the time series. In the autoarima models, it detects 
#automatically non-stationary series and differs by itself. Another basic thing, is to know the number of lags. 

#Autoarima is a really usefull model to do forecast in the short-term.

#Vemos que el mejor modelo que ha elegido es el ARIMA (0, 1, 1), siendo 0 los autoregressive lags, el 1 son las diferencias, pensó que no era estacioanria, 
#y el 1 final es la moving average lag. 
#Por otro lado, analizando los residuos vemos que si ha conseguido ruido blanco. Por lo tanto es una serie estacionaria con la que podemos hacer el forecast..
#Este es más preciso aún, con el 88 por ciento.

# Creamos el modelo y analizamos los residuos
aut_colg <- auto.arima(ts_colgate_train, lambda=0)
summary(aut_colg) # AIC de 209, cuanto más bajo mejor es. 
# Vemos que los residuos se comportan como ruido blanco, media constante en torno a 0. Entonces nuestro modelo está
# ajustado, y podemos realizar predicciones con él. Esto junto al p valor de Ljung-Box test confirman esto. 
checkresiduals(aut_colg)
adf.test(aut_colg$residuals)


# Forecast:
fcast_colg <- forecast(aut_colg, h = 16)
autoplot(fcast_colg)+
  labs(title = "Forecast Colgate 16 weeks")
summary(fcast_colg)

accuracy(aut_colg)


# AUTOARIMA CREST + PRED 16 SEMANAS
#En este caso, elige el modelo ARIMA(3,1,0) como el mejor posible. Si miramos al Mape del 17 %, ese es el error. Asi que nuestro accuracy es del 87 por ciento. 
aut_crest <- auto.arima(ts_crest_train, lambda=0)
summary(aut_crest)
checkresiduals(aut_crest) # AIC DE 26, SE AJUSTA MEJOR AQUÍ
# Vemos que los residuos se comportan como ruido blanco, media constante en torno a 0. Entonces nuestro modelo está
# ajustado, y podemos realizar predicciones con él. Esto junto al p valor de Ljung-Box test confirman esto. 

fcast_crest <- forecast(aut_crest, h = 16)

autoplot(fcast_crest)+
  labs(title = "Forecast Crest 16 weeks")
summary(fcast_crest)
adf.test(aut_crest$residuals)
accuracy(aut_crest)

################################################################################################3
# OUTLIERS Y EFECTOS
# La detección de outliers se hace sencilla con la función tso. 

# COLGATE:
# Vemos que nos coge dos outliers, en 1960 y 1959.
# Es interesante en el plot ver el efecto del primer outlier, que fue una caida brutal en Colgate, propiciada por las medidas de publicidad de crest.
outliers_colg <- tso(y = ts_colgate_train, types = c("AO", "LS", "TC"),
                    discard.method = "bottom-up", 
                    tsmethod = "auto.arima", 
                    args.tsmethod = list(allowdrift = FALSE, ic = "bic"))
plot(outliers_colg)
# Para ver las semanas, son la 102 (semana 50 de 1959) y 136 (semana 32 de 1960).
outliers_idx_colg <- outliers_colg$outliers$ind


# CREST
outliers_crest <- tso(y = ts_crest_train, types = c("AO", "LS", "TC"),
                     discard.method = "bottom-up", 
                     tsmethod = "auto.arima", 
                     args.tsmethod = list(allowdrift = FALSE, ic = "bic"))
plot(outliers_crest)

# Para ver las semanas, son la 136 (semana 32 de 1960), 167 (semana 11 de 1961) y
# 196 (semana 40 de 1961.
outliers_idx_crest <- outliers_crest$outliers$ind

################################################################################################

# MODELOS DE INTERVENCIÓN: ARIMAX (x stands for exogenous variable)
# El modelo de intervención viene a explicar esos datos atípicos de nuestra serie temporal, analizarlos y ver
# qué hacemos con ellos. Antes hemos graficado nuestros outliers y sus efectos. 

# COLGATE 
# A ojo, en la gráfica original se puede ver que el tipo de intervención es  escalón, 
# puesto que cae repentinamente, y se mantiene en esa tendencia decreciente. 
# Una vez detectados los atípicos, podemos realizar el modelo de intervención: 

int_data_col=ts_colgate_train

dummies_colg=data.frame(
  AO1102=1*(seq(ts_colgate_train)==102),
  LS1136=1*(seq(ts_colgate_train)>=136))

# Aquí le metemos otra variable que es la dummies, por eso usamos un ARIMAX. 
mod_int_colg=arimax(int_data_col,order=c(3,0,0),
                   seasonal=list(order=c(1,0,0),period=52),
                   xreg=dummies_colg,
                   method='ML')
mod_int_colg
mod_int_colg$coef


# CREST
# A ojo, en la gráfica original se puede ver que el tipo de inmtervención es intervención escalón, 
# puesto que sube repentinamente, y se mantiene en esa tendencia. 
int_data_crest=ts_crest_train
dummies_crest=data.frame(
  LS1136=1*(seq(ts_crest_train)>=136),
  AO1167=1*(seq(ts_crest_train)==167),
  TC1196=1*(seq(ts_crest_train)>=196))

mod_int_crest=arimax(int_data_crest,order=c(3,1,0),
                   seasonal=list(order=c(1,0,0),period=52),
                   xreg=dummies_crest,
                   method='ML')
mod_int_crest
mod_int_crest$coef



################################################################################################

# FUNCIONES DE TRANSFERENCIA

mod0=dynlm(ts_colgate_train ~ L(ts_crest_train, 0:15) + L(ts_colgate_train, 1))
summary(mod0)

# Modelizacion con ARIMAX
mod_colg_ft <- arimax(ts_colgate_train,
               order=c(1,0,0),
               include.mean=FALSE,
               xtransf=ts_crest_train,
               xreg = ts_crest_train,
               transfer=list(c(0,0)),
               method="ML")

summary(mod_colg_ft)
forecast::tsdisplay(mod_colg_ft$residuals)
mod_colg_ft_fct <- predict(mod_colg_ft, newxreg = ts_crest_test, n.ahead = 16)





