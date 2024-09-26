# Liberias
import pandas as pd 
import matplotlib.pyplot as plt
import ta #para analisis técnico 
from itertools import combinations
import optuna #optimización de parametros

#REalizar operaciones en el codigo de tomar profit o lo q sea
class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares, stop_loss, take_profit):
        self.operation_type = operation_type #Indica si es buy o sell
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.closed = False
        
#Estrategia de trading la segunda clase
#Aqui empezamos con el dinero y las cosas basicas de las comisiones y de cuantas acciones se van a tradear y asi
class TradingStrategy:
    def __init__(self, file):
        self.data = None
        self.operations = []
        self.cash = 1_000_000
        self.com = 0.001
        self.tp = 1.05
        self.sl = .95
        self.strategy_value = [1_000_000]
        self.n_shares = 10
        self.file = file
        
        # SE TIENEN QUE CAMBIAR LOS ARCHIVOS (aapl y btc)
        self.file_mapping = {
            "5m": "data/aapl_5m_train.csv"
        }
        self.load_data(self.file)
        self.indicators = {}
        self.active_indicators = []
        self.calculate_indicators() # para calcular indicadores técnicos 
        self.define_buy_sell_signals() # señales de compra y venta basados en los indicadores
        self.run_signals()
        self.best_combination = None
        self.best_value = 0
        
        #Aqui solo se carga el train y se quita los NA
    def load_data(self, time_frame):
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)
        
         # Indicadores 
    #Aqui se programan los parametros de los indicadores de que las caracteristicas bascias de cada uno y asi

    def calculate_indicators(self):
        # RSI (Relative Strength Index)
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=14)
        self.data['RSI'] = rsi_indicator.rsi()

        # Bandas de Bollinger
        bollinger = ta.volatility.BollingerBands(close=self.data['Close'], window=20, window_dev=2)
        self.data['Bollinger_High'] = bollinger.bollinger_hband()
        self.data['Bollinger_Low'] = bollinger.bollinger_lband()
        self.data['Bollinger_Mid'] = bollinger.bollinger_mavg()

        # Williams %R
        williams_r = ta.momentum.WilliamsRIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], lbp=5)
        self.data['Williams_%R'] = williams_r.williams_r()

        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close=self.data['Close'], window_slow=26, window_fast=12, window_sign=9)
        self.data['MACD'] = macd.macd()
        self.data['Signal_Line'] = macd.macd_signal()

        # Oscilador Estocástico
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=14, smooth_window=3)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal()

        # Elimina valores nulos y resetea el índice
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
     #Aqui se define como se va a llamar cuando cumple la condicion de compro o venta de cada uno
    def define_buy_sell_signals(self):
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'Bollinger': {'buy': self.bollinger_buy_signal, 'sell': self.bollinger_sell_signal},
            'Williams %R': {'buy': self.williams_r_buy_signal, 'sell': self.williams_r_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal},
            'Stoch': {'buy': self.stoch_buy_signal, 'sell': self.stoch_sell_signal}
        }
    
    ###Activacion de indicadores
    #Esta funcion solo dice cuales indicadores estan activos y los agrega a una lista
    def activate_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)
#
#Abajo de esto solo es para saber como se activa cada uno para luego ingresar a la lista anterior
    def stoch_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] < prev_row['stoch_%D'] and row['stoch_%K'] > row['stoch_%D'] and row['stoch_%K'] < 20
    
    def stoch_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] > prev_row['stoch_%D'] and row['stoch_%K'] < row['stoch_%D'] and row['stoch_%K'] > 80

    # Estrategia para RSI
    def rsi_buy_signal(self, row, prev_row=None):
        # Compra cuando RSI esté por debajo de 30 (sobreventa)
        return row.RSI < 30

    def rsi_sell_signal(self, row, prev_row=None):
        # Venta cuando RSI esté por encima de 70 (sobrecompra)
        return row.RSI > 70

    # Estrategia para Bandas de Bollinger
    def bollinger_buy_signal(self, row, prev_row=None):
        # Compra cuando el precio cierre por debajo de la banda inferior
        return row['Close'] < row['Bollinger_Low']

    def bollinger_sell_signal(self, row, prev_row=None):
        # Venta cuando el precio cierre por encima de la banda superior
        return row['Close'] > row['Bollinger_High']

    # Estrategia para MACD
    def macd_buy_signal(self, row, prev_row=None):
        # Compra cuando la línea MACD cruza por encima de la línea de señal
        if prev_row is not None:
            return row.MACD > row.Signal_Line and prev_row.MACD < prev_row.Signal_Line
        return False

    def macd_sell_signal(self, row, prev_row=None):
        # Venta cuando la línea MACD cruza por debajo de la línea de señal
        if prev_row is not None:
            return row.MACD < row.Signal_Line and prev_row.MACD > prev_row.Signal_Line
        return False

    # Estrategia para Williams %R
    def williams_r_buy_signal(self, row, prev_row=None):
        #Compra cuando el Williams %R esté por debajo de -80 (sobreventa)
        return row['Williams_%R'] < -80

    def williams_r_sell_signal(self, row, prev_row=None):
        # Venta cuando el Williams %R esté por encima de -20 (sobrecompra)
        return row['Williams_%R'] > -20

##En esta funcion solo corremos todso los indicadores (Segun yo)
    
    #Correr señales de trade
    def run_signals(self):
    # Ejecutar señales de compra y venta para cada indicador
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data.apply(
                lambda row: self.indicators[indicator]['buy'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1
            )
            self.data[indicator + '_sell_signal'] = self.data.apply(
                lambda row: self.indicators[indicator]['sell'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1
            )

        # Convertir señales de compra y venta a valores numéricos (1 para True, 0 para False)
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data[indicator + '_buy_signal'].astype(int)
            self.data[indicator + '_sell_signal'] = self.data[indicator + '_sell_signal'].astype(int)
    
    #Ejecutar trades
    #Se ejecutan lo de combra y venta si se cumple con los indicadores minimos que son 2
    
    def execute_trades(self, best=False):
        # Verifica si ejecutar con la mejor combinación o con los indicadores activos
        if best:
            for indicator in self.best_combination:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.best_combination]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.best_combination]].sum(axis=1)
            total_active_indicators = len(self.best_combination)
        else:  # Si no es 'best', usar los indicadores activos
            for indicator in self.active_indicators:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.active_indicators]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.active_indicators]].sum(axis=1)
            total_active_indicators = len(self.active_indicators)

        # Iterar sobre cada fila (i.e. cada momento en el tiempo) para ejecutar operaciones basadas en señales
        for i, row in self.data.iterrows():

            # Si hay 2 o menos indicadores activos, necesitas consenso completo para abrir una operación
            if total_active_indicators <= 2:
                if self.data['total_buy_signals'].iloc[i] == total_active_indicators:
                    self._open_operation('long', row)
                elif self.data['total_sell_signals'].iloc[i] == total_active_indicators:
                    self._open_operation('short', row)

            # Si hay más de 2 indicadores activos, se abre operación si la mayoría de señales están activas
            else:
                if self.data['total_buy_signals'].iloc[i] > (total_active_indicators / 2):
                    self._open_operation('long', row)
                elif self.data['total_sell_signals'].iloc[i] > (total_active_indicators / 2):
                    self._open_operation('short', row)

            # Verificar si hay operaciones que deberían cerrarse basadas en stop_loss o take_profit
            self.check_close_operations(row)

            # Actualizar el valor total de la estrategia en cada iteración
            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
            self.strategy_value.append(total_value)

# Operaciones abiertas!! 

#Aqui solo es para las operaciones abiertas el profit y el loss o las condicoines segun yo
    def _open_operation(self, operation_type, row):
            if operation_type == 'long':
                stop_loss = row['Close'] * self.sl
                take_profit = row['Close'] * self.tp
            else:  # 'short'
                stop_loss = row['Close'] * self.sl
                take_profit = row['Close'] * self.tp

            self.operations.append(Operation(operation_type, row['Close'], row.name, self.n_shares, stop_loss, take_profit))
            if operation_type == 'long':
                self.cash -= row['Close'] * self.n_shares * (1 + self.com)
            else:  # 'short'
                self.cash += row['Close'] * self.n_shares * (1 - self.com)  # Incrementa el efectivo al abrir la venta en corto

#print(f"Operación {operation_type} iniciada en {row.name}, Precio: {row['Close']}, Cash restante: {self.cash}")

#Cerrar operaciones
    def check_close_operations(self, row):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= op.take_profit or row['Close'] <= op.stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= op.take_profit or row['Close'] >= op.stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * op.n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * op.n_shares * (1 + self.com)  # Decrementa el efectivo al cerrar la venta en corto, basado en el nuevo precio
                   
                op.closed = True
                #print(f"Operación {op.operation_type} cerrada en {row.name}, Precio: {row['Close']}, Cash resultante: {self.cash}")
    
    def calculate_operation_value(self, op, current_price):
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * op.n_shares if not op.closed else 0
        else:  # 'short'
            return (op.bought_at - current_price) * op.n_shares if not op.closed else 0
#PAra mostrar los resultados
    def plot_results(self, best = False):
        self.reset_strategy()
        if best == False:
            self.execute_trades()
        else:
            self.execute_trades(best=True)
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()
        
    def run_combinations(self):
        all_indicators = list(self.indicators.keys())
        for r in range(1, len(all_indicators) + 1):
            for combo in combinations(all_indicators, r):
                self.active_indicators = list(combo)
                print(f"Ejecutando con combinación de indicadores: {self.active_indicators}")
                self.execute_trades()
                
                final_value = self.strategy_value[-1]
                if final_value > self.best_value:
                    self.best_value = final_value
                    self.best_combination = self.active_indicators.copy()
                self.reset_strategy()

        print(f"Mejor combinación de indicadores: {self.best_combination} con un valor de estrategia de: {self.best_value}")

    def reset_strategy(self):
        self.operations.clear()
        self.cash = 1_000_000
        self.strategy_value = [1_000_000]      
        
#Aqui se optimiza cada parametro de los indicadores    
    #def optimize_parameters(self, study):
        # Aquí es donde pasarías los parámetros optimizados de cada indicador desde el objeto 'study' (ej. Optuna)
        #self.set_rsi_parameters(study.best_params['rsi_window'])
        #self.set_bollinger_parameters(study.best_params['bollinger_window'], study.best_params['bollinger_std_dev'])
        #self.set_williams_r_parameters(study.best_params['williams_r_window'])
        #self.set_macd_parameters(study.best_params['macd_fast'], study.best_params['macd_slow'], study.best_params['macd_sign'])
        #self.set_stoch_parameters(study.best_params['stoch_k_window'], study.best_params['stoch_d_window'], study.best_params['stoch_smoothing'])
    
    def optimize_parameters(self):
        def objective(trial):
            self.reset_strategy()
            # Configura los parámetros para cada indicador activo en la mejor combinación
            for indicator in self.best_combination:
                if indicator == 'RSI':
                    rsi_window = trial.suggest_int('rsi_window', 5, 30)
                    self.set_rsi_parameters(rsi_window)
                elif indicator == 'Bollinger':
                    bollinger_window = trial.suggest_int('bollinger_window', 10, 50)
                    bollinger_std = trial.suggest_float('bollinger_std', 1, 3)
                    self.set_bollinger_parameters(bollinger_window, bollinger_std)
                elif indicator == 'Williams %R':
                    williams_r_period = trial.suggest_int('williams_r_period', 5, 30)
                    self.set_williams_r_parameters(williams_r_period)
                elif indicator == 'MACD':
                    macd_fast = trial.suggest_int('macd_fast', 10, 20)
                    macd_slow = trial.suggest_int('macd_slow', 21, 40)
                    macd_sign = trial.suggest_int('macd_sign', 5, 15)
                    self.set_macd_parameters(macd_fast, macd_slow, macd_sign)
                elif indicator == 'Stoch':
                    stoch_k_window = trial.suggest_int('stoch_k_window', 5, 21)
                    stoch_d_window = trial.suggest_int('stoch_d_window', 3, 14)
                    stoch_smoothing = trial.suggest_int('stoch_smoothing', 3, 14)
                    self.set_stoch_parameters(stoch_k_window, stoch_d_window, stoch_smoothing)

            # Configuración de take profit, stop loss y n_shares
            tp= trial.suggest_float('take_profit', 1.01, 1.5)
            sl = trial.suggest_float('stop_loss', 0.5, 0.9)
            n_shares = trial.suggest_int('n_shares', 1, 100)

            # Asumimos que existe una función para ajustar estos parámetros en la estrategia
            self.set_trade_parameters(tp, sl, n_shares)

            # Ejecutar la estrategia con la mejor combinación y los nuevos parámetros
            self.run_signals()
            self.execute_trades(best=True)
    
            return self.strategy_value[-1]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)  # Ajusta el número de pruebas según sea necesario

    # Imprimir y aplicar los mejores parámetros encontrados para cada indicador
        print(f"Mejores parámetros encontrados: {study.best_params}")
        for indicator in self.best_combination:
            if indicator == 'RSI':
                self.set_rsi_parameters(study.best_params['rsi_window'])
            elif indicator == 'Bollinger':
                self.set_bollinger_parameters(study.best_params['bollinger_window'], study.best_params['bollinger_std'])
            elif indicator == 'Williams %R':
                self.set_williams_r_parameters(study.best_params['williams_r_period'])
            elif indicator == 'MACD':
                self.set_macd_parameters(study.best_params['macd_fast'], study.best_params['macd_slow'], study.best_params['macd_sign'])
            elif indicator == 'Stoch':
                self.set_stoch_parameters(study.best_params['stoch_k_window'], study.best_params['stoch_d_window'], study.best_params['stoch_smoothing'])

        # Aquí también debes aplicar los parámetros de take_profit, stop_loss y n_shares optimizados
        self.set_trade_parameters(study.best_params['tp'], study.best_params['sl'], study.best_params['n_shares'])

    
    
    # Parámetros RSI
    def set_rsi_parameters(self, window):
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=window)
        self.data['RSI'] = rsi_indicator.rsi()

    # Parámetros Bandas de Bollinger
    def set_bollinger_parameters(self, window, std_dev):
        bollinger = ta.volatility.BollingerBands(close=self.data['Close'], window=window, window_dev=std_dev)
        self.data['Bollinger_High'] = bollinger.bollinger_hband()
        self.data['Bollinger_Low'] = bollinger.bollinger_lband()
        self.data['Bollinger_Mid'] = bollinger.bollinger_mavg()

    # Parámetros Williams %R
    def set_williams_r_parameters(self, window):
       williams_r = ta.momentum.WilliamsRIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], lbp=window)
       self.data['Williams_%R'] = williams_r.williams_r()

    # Parámetros MACD
    def set_macd_parameters(self, fast, slow, sign):
        macd = ta.trend.MACD(close=self.data['Close'], window_slow=slow, window_fast=fast, window_sign=sign)
        self.data['MACD'] = macd.macd()
        self.data['Signal_Line'] = macd.macd_signal()

    # Parámetros Oscilador Estocástico
    def set_stoch_parameters(self, k_window, d_window, smoothing):
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=k_window, smooth_window=d_window)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal().rolling(window=smoothing).mean()
    
    
    ####SE TIENEN QUE CAMBIAR LOS ARCHIVOS POR LOS QUE TENEMOS NOSOTROS
    def test(self):
        test_file_mapping = {
            "5m": "data/aapl_5m_test.csv"
        }
        self.load_data(self.file)
        self.calculate_indicators()
        self.define_buy_sell_signals()
        self.run_signals()
        self.execute_trades(best=True)
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()        

#Despues de aqui abajo solo es para hacer graficas mostrando los resultados en el archivo ya bonito

    def show_ClosingPrices(self):
        plt.figure(figsize=(12, 6))  # Tamaño del gráfico
        plt.plot(self.data.index[:214], self.data['Close'][:214], label='Close Price')  # Gráfico de la serie de precios de cierre
        plt.title('Closing Prices')  # Título del gráfico
        plt.xlabel('Index')  # Etiqueta del eje X
        plt.ylabel('Price')  # Etiqueta del eje Y
        plt.legend()  # Añadir leyenda
        plt.show()  # Mostrar el gráfico

        
    def show_RSI(self):
        plt.figure(figsize=(12, 6))  # Tamaño del gráfico
        plt.plot(self.data.index[:214], self.data['RSI'][:214], label='RSI')  # Gráfico del RSI
        plt.axhline(70, color='red', linestyle='--', label="Overbought (70)")  # Línea para sobrecompra
        plt.axhline(30, color='green', linestyle='--', label="Oversold (30)")  # Línea para sobreventa
        plt.title('RSI Indicator')  # Título del gráfico
        plt.xlabel('Index')  # Etiqueta del eje X
        plt.ylabel('RSI Value')  # Etiqueta del eje Y
        plt.legend()  # Añadir leyenda
        plt.show()  # Mostrar el gráfico


    def show_bollinger_bands(self):
        plt.figure(figsize=(12, 6))

        # Plot closing prices
        plt.plot(self.data.index[:214], self.data['Close'][:214], label='Close Price', color='blue')

        # Plot Bollinger Bands
        plt.plot(self.data.index[:214], self.data['Bollinger_High'][:214], label='Upper Bollinger Band', linestyle='--', color='red')
        plt.plot(self.data.index[:214], self.data['Bollinger_Low'][:214], label='Lower Bollinger Band', linestyle='--', color='green')
        plt.plot(self.data.index[:214], self.data['Bollinger_Mid'][:214], label='Middle Band', linestyle='--', color='black')

        plt.fill_between(self.data.index[:214], self.data['Bollinger_Low'][:214], self.data['Bollinger_High'][:214], color='gray', alpha=0.3)

        plt.title('Bollinger Bands with Closing Prices')
        plt.legend()
        plt.show()


    def show_williams_r(self):
        plt.figure(figsize=(12, 6))

        plt.plot(self.data.index[:214], self.data['Williams_%R'][:214], label="Williams %R", color='purple')
        plt.axhline(-20, color='red', linestyle='--', label="Overbought (-20)")
        plt.axhline(-80, color='green', linestyle='--', label="Oversold (-80)")

        plt.title('Williams %R Indicator')
        plt.legend()
        plt.show()


    def show_MACD(self):
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line']

        plt.figure(figsize=(12, 6))

        # Plot the MACD and the signal line
        plt.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        plt.plot(self.data.index, self.data['Signal_Line'], label='Signal_Line', color='red')

        # Fill the histogram between MACD and the signal line
        # We will use a different color depending on whether the histogram is positive or negative
        plt.bar(self.data.index, self.data['MACD_Histogram'], label='MACD Histogram', color=['green' if val >= 0 else 'red' for val in self.data['MACD_Histogram']])

        # Add title and legend
        plt.title('MACD, Signal Line, and Histogram')
        plt.legend()
        plt.xlim(0, 214)
        plt.ylim(-1, 1)
        plt.show()

        
    def show_stochastic(self):
        
        plt.figure(figsize=(12, 6))  # Tamaño del gráfico
        plt.plot(self.data.index[:250], self.data['stoch_%K'][:250], label='%K', color='blue')  # Línea del %K
        plt.plot(self.data.index[:250], self.data['stoch_%D'][:250], label='%D', color='orange')  # Línea del %D
        plt.axhline(80, color='red', linestyle='--', label="Overbought (80)")  # Línea de sobrecompra
        plt.axhline(20, color='green', linestyle='--', label="Oversold (20)")  # Línea de sobreventa
        plt.title('Stochastic Oscillator')  # Título del gráfico
        plt.xlabel('Index')  # Etiqueta del eje X
        plt.ylabel('Stochastic Value')  # Etiqueta del eje Y
        plt.legend()  # Leyenda
        plt.show()  # Mostrar el gráfico
        
     #Editar las cosas para nuestro archivo   
    def rendimiento(data_path, cash=1000000):
        # Leer el archivo CSV
        data = pd.read_csv(data_path)

        # Convertir la columna 'Date' a tipo datetime
        data['Date'] = pd.to_datetime(data['Date'])

        # Obtener el precio de cierre del primer y último dato
        primer_cierre = data.iloc[0]['Close']
        ultimo_cierre = data.iloc[-1]['Close']

        # Calcular el rendimiento del activo
        rend_pasivo = (ultimo_cierre - primer_cierre) / primer_cierre
        print("The passive asset return from the first close to the last close is: {:.2%}".format(rend_pasivo))

        # Comparativa con la estrategia utilizada
        cashfinal = 1107153.05  # Modifica este valor según lo que obtengas
        rend_estrategia = (cashfinal - cash) / cash
        print("The strategy return from the first close to the last close is: {:.2%}".format(rend_estrategia))

        # Ordenar los datos por fecha si no están ordenados
        data = data.sort_values(by='Date')

        # Graficar el precio de cierre del activo
        plt.figure(figsize=(12, 8))
        plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        plt.title('Close Price of the Asset')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calcular el rendimiento acumulado
        data['Returns'] = data['Close'].pct_change().fillna(0)

        # Calcular el valor acumulado
        initial_investment = cash
        data['Investment_Value'] = (1 + data['Returns']).cumprod() * initial_investment

        # Graficar el rendimiento de la inversión
        plt.figure(figsize=(12, 8))
        plt.plot(data['Date'], data['Investment_Value'], label='Investment Value', color='green')
        plt.title('Investment Return')
        plt.xlabel('Date')
        plt.ylabel('Investment Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        valor_final = data['Investment_Value'].iloc[-1]
        print("The final value of the investment: ${:,.2f}".format(valor_final))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        