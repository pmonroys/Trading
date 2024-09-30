import pandas as pd 
import matplotlib.pyplot as plt
import ta  
from itertools import combinations
import optuna 


# This class is used to represent a trade operation, either a buy (long) or sell (short), and stores the trade's details like price, stop-loss, and take-profit.
class Operation:    
    
    def __init__(self, operation_type, bought_at, timestamp, n_shares, stop_loss, take_profit):
        self.operation_type = operation_type # Type of operation: 'long' for buy or 'short' for sell
        self.bought_at = bought_at           # The price at which the trade was made
        self.timestamp = timestamp           # The timestamp of the trade
        self.n_shares = n_shares             # The number of shares involved in the trade
        self.stop_loss = stop_loss           # Stop-loss price to limit losses
        self.take_profit = take_profit       # Take-profit price to lock in gains
        self.closed = False                  # Indicates if the trade is still open or closed
        
        
# This class is used to load data, calculate technical indicators, and perform trading strategies based on the calculated indicators.
class TechnicalAnalysis:    
    
    #Initializes the class and sets up the main parameters for the trading strategy.
    def __init__(self, file):
        self.data = None                               # Placeholder for the loaded data
        self.operations = []                           # List of trade operations
        self.cash = 1_000_000                          # Starting cash amount for the strategy
        self.com = 0.001                               # Commission fee for each trade
        self.strategy_value = [1_000_000]              # Strategy value starting at 1M
        self.n_shares = 1/10                           # Default number of shares per trade
        self.file = file                               # File to load data from
        self.stop_loss_percentage = 0.05               # Stop loss percentage
        self.take_profit_percentage = 0.10             # Take profit percentage
        self.file_mapping = {                          # Mapping between assets and their CSV files
            "APPLE": "data/aapl_5m_train.csv",
            "BITCOIN": "data/btc_project_train.csv"
        }
        self.load_data(self.file)                      # Load data from the file
        self.indicators = {}                           # Dictionary for storing indicators
        self.active_indicators = []                    # List of active indicators
        self.calculate_indicators()                    # Call to calculate technical indicators
        self.define_buy_sell_signals()                 # Define the buy/sell signals
        self.run_signals()                             # Apply signals to the data
        self.best_combination = None                   # Best indicator combination found
        self.best_value = 100000                       # Best strategy value
  

    # Loads the CSV data for the given asset type.    
    def load_data(self, asset_type):
        file_name = self.file_mapping.get(asset_type)  # Get the file name from the mapping
        if not file_name:
            raise ValueError(f"Asset type '{asset_type}' not found in file mapping.") # Raise an error if asset type is invalid
        self.data = pd.read_csv(file_name)             # Load the CSV file
        self.data.dropna(inplace=True)                 # Remove missing values


    #This function calculates various technical indicators like RSI, Bollinger Bands, MACD, etc., and stores them in the self.data DataFrame.
    def calculate_indicators(self):
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=14) # Calculate RSI with a 14-period window
        self.data['RSI'] = rsi_indicator.rsi()                                        # Add RSI values to the data
        
        bollinger = ta.volatility.BollingerBands(close=self.data['Close'], window=20, window_dev=2) # Calculate Bollinger Bands
        self.data['Bollinger_High'] = bollinger.bollinger_hband()                     # Upper Bollinger Band
        self.data['Bollinger_Low'] = bollinger.bollinger_lband()                      # Lower Bollinger Band
        self.data['Bollinger_Mid'] = bollinger.bollinger_mavg()                       # Middle Bollinger Band

        williams_r = ta.momentum.WilliamsRIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], lbp=5) # Williams %R indicator
        self.data['Williams_%R'] = williams_r.williams_r()

        macd = ta.trend.MACD(close=self.data['Close'], window_slow=26, window_fast=12, window_sign=9) # Calculate MACD
        self.data['MACD'] = macd.macd()                                               # MACD line
        self.data['Signal_Line'] = macd.macd_signal()                                 # Signal line

        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=14, smooth_window=3) # Stochastic Oscillator
        self.data['stoch_%K'] = stoch_indicator.stoch()                               # %K line
        self.data['stoch_%D'] = stoch_indicator.stoch_signal()                        # %D line

        self.data.dropna(inplace=True)                                                # Remove missing values after adding indicators
        self.data.reset_index(drop=True, inplace=True)                                # Reset the index after cleaning the data
         
            
    # Defines the buy and sell signals for each technical indicator.
    def define_buy_sell_signals(self):
        self.indicators = {  # Define the buy/sell logic for each indicator
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'Bollinger': {'buy': self.bollinger_buy_signal, 'sell': self.bollinger_sell_signal},
            'Williams %R': {'buy': self.williams_r_buy_signal, 'sell': self.williams_r_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal},
            'Stoch': {'buy': self.stoch_buy_signal, 'sell': self.stoch_sell_signal}
        }

        
    # Activates a specified technical indicator by adding it to the list of active indicators.    
    def activate_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)

            
    # Defines the buy/sell conditions for the Stochastic Oscillator.       
    def stoch_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] < prev_row['stoch_%D'] and row['stoch_%K'] > row['stoch_%D'] and row['stoch_%K'] < 20
    # Buy when %K crosses above %D and %K is below 20 (oversold)
       
    def stoch_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] > prev_row['stoch_%D'] and row['stoch_%K'] < row['stoch_%D'] and row['stoch_%K'] > 80
    # Sell when %K crosses below %D and %K is above 80 (overbought)


    # Defines the buy/sell conditions based on the Relative Strength Index (RSI).
    def rsi_buy_signal(self, row, prev_row=None):
        return row.RSI < 30 # Buy when RSI is below 30 (oversold)
    
    def rsi_sell_signal(self, row, prev_row=None):
        return row.RSI > 70 # Sell when RSI is above 70 (overbought)

    
    # Defines the buy/sell conditions based on Bollinger Bands.
    def bollinger_buy_signal(self, row, prev_row=None):
        return row['Close'] < row['Bollinger_Low'] # Buy when price is below the lower Bollinger band
    
    def bollinger_sell_signal(self, row, prev_row=None):
        return row['Close'] > row['Bollinger_High'] # Sell when price is above the upper Bollinger band

    
    # Defines the buy/sell conditions based on the Moving Average Convergence Divergence (MACD).
    def macd_buy_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD > row.Signal_Line and prev_row.MACD < prev_row.Signal_Line # Buy when MACD crosses above the signal line
        return False
  
    def macd_sell_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD < row.Signal_Line and prev_row.MACD > prev_row.Signal_Line # Sell when MACD crosses below the signal line
        return False

    
    # Defines the buy/sell conditions based on the Williams %R indicator.
    def williams_r_buy_signal(self, row, prev_row=None):
        return row['Williams_%R'] < -80 # Buy when Williams %R is below -80 (oversold)
    
    def williams_r_sell_signal(self, row, prev_row=None):
        return row['Williams_%R'] > -20 # Sell when Williams %R is above -20 (overbought)

    
    #This function calculates the maximum drawdown of the strategy. Drawdown is the peak-to-trough decline in the value of the strategy.
    def calculate_max_drawdown(self):                                                    
        cumulative_returns = (1 + pd.Series(self.strategy_value).pct_change()).cumprod() # Calculate cumulative returns
        peak = cumulative_returns.cummax()                                               # Track the highest value (peak)
        drawdown = (peak - cumulative_returns) / peak                                    # Calculate drawdown as the percentage drop from the peak                 
        max_drawdown = drawdown.max()                                                    # Get the maximum drawdown value
        return max_drawdown

    
    #This function applies the buy and sell signals for all indicators and stores the results in the DataFrame.
    def run_signals(self):
        for indicator in list(self.indicators.keys()): # For each indicator
            self.data[indicator + '_buy_signal'] = self.data.apply(
                lambda row: self.indicators[indicator]['buy'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1
            ) # Apply the buy signal for each row
            self.data[indicator + '_sell_signal'] = self.data.apply(
                lambda row: self.indicators[indicator]['sell'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1
            ) # Apply the sell signal for each row

        for indicator in list(self.indicators.keys()): # Convert the signals to integer (1 for True, 0 for False)
            self.data[indicator + '_buy_signal'] = self.data[indicator + '_buy_signal'].astype(int)
            self.data[indicator + '_sell_signal'] = self.data[indicator + '_sell_signal'].astype(int)
        
        
    # This function executes trades based on the signals generated by the active indicators.    
    def execute_trades(self, best=False):
        if best: # If using the best combination of indicators
            for indicator in self.best_combination:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.best_combination]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.best_combination]].sum(axis=1)
            total_active_indicators = len(self.best_combination)
        else: # If using active indicators
            for indicator in self.active_indicators:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.active_indicators]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.active_indicators]].sum(axis=1)
            total_active_indicators = len(self.active_indicators)

        for i, row in self.data.iterrows(): # Iterate through the data and check for buy/sell signals
            if total_active_indicators <= 2: # If 2 or fewer indicators are active
                if self.data['total_buy_signals'].iloc[i] == total_active_indicators:
                    self._open_operation('long', row) # Open a long position
                elif self.data['total_sell_signals'].iloc[i] == total_active_indicators:
                    self._open_operation('short', row) # Open a short position
            else: # If more than 2 indicators are active
                if self.data['total_buy_signals'].iloc[i] > (total_active_indicators / 2):
                    self._open_operation('long', row) # Open a long position if the majority of signals are buy
                elif self.data['total_sell_signals'].iloc[i] > (total_active_indicators / 2):
                    self._open_operation('short', row) # Open a short position if the majority of signals are sell
            
            self.check_close_operations(row) # Check if any open operations should be closed

            # Update the total strategy value (cash + value of open positions)
            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
            self.strategy_value.append(total_value)

            
    # Opens a long or short trade based on the signals generated.       
    def _open_operation(self, operation_type, row):
        if operation_type == 'long': # Long trade: set stop loss and take profit
            stop_loss = row['Close'] * (1 - self.stop_loss_percentage)  
            take_profit = row['Close'] * (1 + self.take_profit_percentage) 
        else: # Short trade: set stop loss and take profit  
            stop_loss = row['Close'] * (1 + self.stop_loss_percentage)
            take_profit = row['Close'] * (1 - self.take_profit_percentage)
      
        self.operations.append(Operation(operation_type, row['Close'], row.name, self.n_shares, stop_loss, take_profit))

        if operation_type == 'long': # For a long position, subtract the cost of the trade from cash
            self.cash -= row['Close'] * self.n_shares * (1 + self.com)
        else: # For a short position, add the value of the trade to cash  
            self.cash += row['Close'] * self.n_shares * (1 - self.com)

   
    #This function checks if any of the open operations should be closed based on stop-loss or take-profit conditions.
    def check_close_operations(self, row):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= op.take_profit or row['Close'] <= op.stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= op.take_profit or row['Close'] >= op.stop_loss))):
                if op.operation_type == 'long': # If long, sell at the current price
                    self.cash += row['Close'] * op.n_shares * (1 - self.com)
                else: # If short, buy back at the current price   
                    self.cash -= row['Close'] * op.n_shares * (1 + self.com)
                   
                op.closed = True # Mark the operation as closed
             
            
    # Calculates the value of a given operation based on the current market price.        
    def calculate_operation_value(self, op, current_price):
        if op.operation_type == 'long': # For long trades, profit is current price minus buy price
            return (current_price - op.bought_at) * op.n_shares if not op.closed else 0
        else: # For short trades, profit is buy price minus current price  
            return (op.bought_at - current_price) * op.n_shares if not op.closed else 0
        
        
    # This function tests different combinations of indicators to find the best set that minimizes the strategy's drawdown. It tries combinations of 1 indicator, 2 indicators, and so on.   
    def run_combinations(self):
        all_indicators = list(self.indicators.keys()) # Get a list of all available indicators
        for r in range(1, len(all_indicators) + 1): # Loop through combinations of indicators, starting from 1 to the total number of indicators
            for combo in combinations(all_indicators, r): # Generate all possible combinations of `r` indicators
                self.active_indicators = list(combo) # Set the active indicators to the current combination
                print(f"Ejecutando con combinación de indicadores: {self.active_indicators}") # Print the current combination being tested
                self.execute_trades() # Execute trades based on the active indicators
                
                max_drawdown = self.calculate_max_drawdown() # Calculate the drawdown for the current strategy
                if max_drawdown < self.best_value: # If this combination has a smaller drawdown, update the best combination
                    self.best_value = max_drawdown
                    self.best_combination = self.active_indicators.copy() # Store the best combination of indicators
                
        print(f"Mejor combinación de indicadores: {self.best_combination} con un valor de estrategia de: {self.best_value}") # Print the best combination of indicators and its drawdown value
 

    # This function runs the trading strategy using the best combination of indicators manually. It allows you to avoid running the full run_combinations() process again, which can be time-consuming. Instead, you can directly input the best combination of indicators to quickly test or run the strategy.
    def run_best(self):  
        self.active_indicators = ['RSI'] # Sets the best combination of indicators manually (e.g., RSI here, but should be updated based on prior runs)
        print(f"Ejecutando con combinación de indicadores: {self.active_indicators}") # Print the combination being used for execution
        self.execute_trades() # Execute trades using the selected indicators
    
        max_drawdown = self.calculate_max_drawdown() # Calculate the maximum drawdown
    
        if max_drawdown < self.best_value: # If the current drawdown is lower than the best recorded drawdown  
            self.best_value = max_drawdown # Update the best value
            self.best_combination = self.active_indicators.copy() # Save the best combination of indicators

        print(f"Mejor combinación de indicadores: {self.best_combination} con un Max Drawdown de: {self.best_value}") # Print the final best combination of indicators and its drawdown

        
    # This function resets the trading strategy by clearing all previous operations and resetting the cash balance and strategy value to the initial starting amounts. It's useful when you want to run a fresh test of the strategy from scratch.
    def reset_strategy(self):
        self.operations.clear()            # Clears the list of all previous trading operations
        self.cash = 1_000_000              # Resets cash to the starting value of 1 million
        self.strategy_value = [1_000_000]  # Resets the strategy value to the initial amount    
    
    
    # This function uses the Optuna library to optimize key parameters of the trading strategy (such as the number of shares, stop-loss percentage, take-profit percentage, and specific indicator parameters). The goal is to minimize the maximum drawdown through parameter optimization.
    def optimize_parameters(self):
        def objective(trial): # This is the objective function to minimize
            self.reset_strategy() # Reset the strategy for each trial       
            self.n_shares = trial.suggest_float('n_shares', 1/100, 1/10) # Suggest a number of shares
            
            stop_loss_percentage = trial.suggest_float('stop_loss', 0.01, 0.2) # Suggest stop-loss percentage   
            take_profit_percentage = trial.suggest_float('take_profit', 0.05, 0.5) # Suggest take-profit percentage 
            
            self.stop_loss_percentage = stop_loss_percentage # Set the suggested stop-loss percentage
            self.take_profit_percentage = take_profit_percentage # Set the suggested take-profit percentage
            
            for indicator in self.best_combination: # Loop over the best indicator combination and optimize parameters
                if indicator == 'RSI':
                    rsi_window = trial.suggest_int('rsi_window', 5, 30) # Suggest RSI window between 5 and 30 periods
                    self.set_rsi_parameters(rsi_window)
                elif indicator == 'Bollinger':
                    bollinger_window = trial.suggest_int('bollinger_window', 10, 50) # Suggest Bollinger Bands window between 10 and 50
                    bollinger_std = trial.suggest_float('bollinger_std', 1, 3) # Suggest the standard deviation multiplier for Bollinger Bands
                    self.set_bollinger_parameters(bollinger_window, bollinger_std)
                elif indicator == 'Williams %R':
                    williams_r_period = trial.suggest_int('williams_r_period', 5, 30) # Suggest the Williams %R period
                    self.set_williams_r_parameters(williams_r_period)
                elif indicator == 'MACD':
                    macd_fast = trial.suggest_int('macd_fast', 10, 20) # Suggest MACD fast period
                    macd_slow = trial.suggest_int('macd_slow', 21, 40) # Suggest MACD slow period
                    macd_sign = trial.suggest_int('macd_sign', 5, 15) # Suggest MACD signal period
                    self.set_macd_parameters(macd_fast, macd_slow, macd_sign)
                elif indicator == 'Stoch':
                    stoch_k_window = trial.suggest_int('stoch_k_window', 5, 21) # Suggest Stochastic K window
                    stoch_d_window = trial.suggest_int('stoch_d_window', 3, 14) # Suggest Stochastic D window
                    stoch_smoothing = trial.suggest_int('stoch_smoothing', 3, 10) # Suggest smoothing window for Stochastic D
                    self.set_stoch_parameters(stoch_k_window, stoch_d_window, stoch_smoothing)

            self.execute_trades() # Execute trades with the suggested parameters
    
            max_drawdown = self.calculate_max_drawdown() # Calculate maximum drawdown for the current trial
            return max_drawdown # Minimize the maximum drawdown  

        study = optuna.create_study(direction='minimize') # Create an Optuna study aiming to minimize the objective (max drawdown)
        study.optimize(objective, n_trials=100) # Run the optimization for n trials

        print(f"Parámetros óptimos: {study.best_params}") # Print the best parameters found
        for indicator in self.best_combination: # Set the optimized parameters for the best combination of indicators
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
        self.n_shares = study.best_params['n_shares'] # Set the best number of shares
    
    
    # This function sets the window size for calculating the RSI (Relative Strength Index) and recalculates the RSI with the new parameters.
    def set_rsi_parameters(self, window):
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=window) # Create an RSI indicator with the specified window size
        self.data['RSI'] = rsi_indicator.rsi() # Calculate and store the new RSI values

    
    # This function sets the window size and standard deviation multiplier for calculating the Bollinger Bands and recalculates the bands with the new parameters.
    def set_bollinger_parameters(self, window, std_dev):
        bollinger = ta.volatility.BollingerBands(close=self.data['Close'], window=window, window_dev=std_dev) # Create Bollinger Bands with the specified window and standard deviation
        self.data['Bollinger_High'] = bollinger.bollinger_hband() # Upper Bollinger Band
        self.data['Bollinger_Low'] = bollinger.bollinger_lband()  # Lower Bollinger Band
        self.data['Bollinger_Mid'] = bollinger.bollinger_mavg()   # Middle Bollinger Band

        
    # This function sets the lookback period for calculating the Williams %R indicator and recalculates it with the new parameters.    
    def set_williams_r_parameters(self, window):
        williams_r = ta.momentum.WilliamsRIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], lbp=window) # Create Williams %R with the specified window
        self.data['Williams_%R'] = williams_r.williams_r() # Calculate and store the new Williams %R values

        
    # This function sets the fast, slow, and signal line windows for calculating the MACD (Moving Average Convergence Divergence) and recalculates it with the new parameters.    
    def set_macd_parameters(self, fast, slow, sign):
        macd = ta.trend.MACD(close=self.data['Close'], window_slow=slow, window_fast=fast, window_sign=sign) # Create MACD with specified fast, slow, and signal window
        self.data['MACD'] = macd.macd() # Calculate and store the MACD line
        self.data['Signal_Line'] = macd.macd_signal() # Calculate and store the signal line 

        
    def set_stoch_parameters(self, k_window, d_window, smoothing):
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=k_window, smooth_window=d_window) # Create MACD with specified fast, slow, and signal window
        self.data['stoch_%K'] = stoch_indicator.stoch() # Calculate and store the MACD line
        self.data['stoch_%D'] = stoch_indicator.stoch_signal().rolling(window=smoothing).mean() # Calculate and store the signal line     


    def scaled_returns(self):
        returns = self.data['Close'].pct_change() # Calculate daily returns based on the closing prices
        initial_investment = 1_000_000
        self.data['Investment_Value'] = initial_investment * (1 + returns).cumprod()  # Compute the cumulative investment value
        scaled_returns = self.data['Investment_Value'] 

        return scaled_returns
    
    
    # Max Drawdown for the passive strategy
    def passive_max_drawdown(self):
        cumulative_returns =  (1 + pd.Series(self.scaled_returns).pct_change()).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (peak - cumulative_returns) / peak  # Calculate drawdown as the relative difference from the peak
        max_drawdown = drawdown.max()
        print(f"En estrategía pasiva el Max Drawdown es de: {max_drawdown}")
        return max_drawdown
    
    
    # Plot_results
    def plot_results(self): 
        self.scaled_returns()
        self.scaled_returns = self.scaled_returns()
        plt.figure(figsize=(12, 8))

        plt.plot(self.strategy_value, label='Strategy Value')
        plt.plot(self.scaled_returns,label='Passive Strategy Value') 

        plt.xlabel('Number of Trades or Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show() 
        
        
    # This function plots the closing prices of the asset.
    def plot_ClosingPrices(self):
        plt.figure(figsize=(12, 6)) # Set the figure size  
        plt.plot(self.data.index[:214], self.data['Close'][:214], label='Close Price') # Plot the closing prices for the first 214 points  
        plt.title('Closing Prices') # Add a title to the plot  
        plt.xlabel('Index') # Label the x-axis  
        plt.ylabel('Price') # Label the y-axis  
        plt.legend() # Add a legend  
        plt.show() # Display the plot  

        
    # This function plots the RSI (Relative Strength Index) values and highlights the overbought (70) and oversold (30) levels.    
    def plot_RSI(self):
        plt.figure(figsize=(12, 6)) # Set the figure size  
        plt.plot(self.data.index[:214], self.data['RSI'][:214], label='RSI') # Plot the RSI values for the first 214 points  
        plt.axhline(70, color='red', linestyle='--', label="Overbought (70)") # Draw a horizontal line at 70 (overbought level)  
        plt.axhline(30, color='green', linestyle='--', label="Oversold (30)") # Draw a horizontal line at 30 (oversold level)  
        plt.title('RSI Indicator') # Add a title to the plot 
        plt.xlabel('Index') # Label the x-axis  
        plt.ylabel('RSI Value') # Label the y-axis  
        plt.legend() # Add a legend  
        plt.show() # Display the plot  

        
    # This function plots the closing prices along with the Bollinger Bands (upper, lower, and middle bands).
    def plot_bollinger_bands(self):
        plt.figure(figsize=(12, 6)) # Set the figure size        
        plt.plot(self.data.index[:214], self.data['Close'][:214], label='Close Price', color='blue') # Plot the closing prices
        plt.plot(self.data.index[:214], self.data['Bollinger_High'][:214], label='Upper Bollinger Band', linestyle='--', color='red') # Plot upper Bollinger Band
        plt.plot(self.data.index[:214], self.data['Bollinger_Low'][:214], label='Lower Bollinger Band', linestyle='--', color='green') # Plot lower Bollinger Band
        plt.plot(self.data.index[:214], self.data['Bollinger_Mid'][:214], label='Middle Band', linestyle='--', color='black') # Plot middle Bollinger Band
        plt.fill_between(self.data.index[:214], self.data['Bollinger_Low'][:214], self.data['Bollinger_High'][:214], color='gray', alpha=0.3) # Fill the area between the upper and lower bands
        plt.title('Bollinger Bands with Closing Prices') # Add a title to the plot
        plt.legend() # Add a legend
        plt.show() # Display the plot

        
    # This function plots the Williams %R indicator values and highlights the overbought (-20) and oversold (-80) levels.
    def plot_williams_r(self):
        plt.figure(figsize=(12, 6)) # Set the figure size
        plt.plot(self.data.index[:214], self.data['Williams_%R'][:214], label="Williams %R", color='purple') # Plot the Williams %R values
        plt.axhline(-20, color='red', linestyle='--', label="Overbought (-20)") # Draw a horizontal line at -20 (overbought level)
        plt.axhline(-80, color='green', linestyle='--', label="Oversold (-80)") # Draw a horizontal line at -80 (oversold level)
        plt.title('Williams %R Indicator') # Add a title to the plot
        plt.legend() # Add a legend
        plt.show() # Display the plot
        
        
    # This function plots the MACD line, signal line, and the MACD histogram.    
    def plot_MACD(self):
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line'] # Calculate the MACD histogram (MACD - Signal Line)
        plt.figure(figsize=(12, 6)) # Set the figure size
        plt.plot(self.data.index, self.data['MACD'], label='MACD', color='blue') # Plot the MACD line in blue
        plt.plot(self.data.index, self.data['Signal_Line'], label='Signal_Line', color='red') # Plot the Signal Line in red
        plt.bar(self.data.index, self.data['MACD_Histogram'], label='MACD Histogram', color=['green' if val >= 0 else 'red' for val in self.data['MACD_Histogram']]) # Plot the MACD histogram as a bar chart
        plt.title('MACD, Signal Line, and Histogram') # Add a title to the plot
        plt.legend() # Add a legend
        plt.xlim(0, 214) # Set x-axis limits
        plt.ylim(-2, 2) # Set y-axis limits
        plt.show() # Display the plot

        
    # This function plots the Stochastic Oscillator (%K and %D lines) and highlights the overbought (80) and oversold (20) levels.
    def plot_stochastic(self):        
        plt.figure(figsize=(12, 6)) # Set the figure size  
        plt.plot(self.data.index[:250], self.data['stoch_%K'][:250], label='%K', color='blue') # Plot the %K line  
        plt.plot(self.data.index[:250], self.data['stoch_%D'][:250], label='%D', color='orange') # Plot the %D line  
        plt.axhline(80, color='red', linestyle='--', label="Overbought (80)") # Draw a horizontal line at 80 (overbought level)  
        plt.axhline(20, color='green', linestyle='--', label="Oversold (20)") # Draw a horizontal line at 20 (oversold level) 
        plt.title('Stochastic Oscillator') # Add a title to the plot  
        plt.xlabel('Index') # Label the x-axis   
        plt.ylabel('Stochastic Value') # Label the y-axis  
        plt.legend() # Add a legend   
        plt.show() # Display the plot
        
        

                                                      
        
class TestStrategy:
    def __init__(self, file):
        self.data = None
        self.operations = []
        self.cash = 1_000_000
        self.com = 0.001
        self.strategy_value = [1_000_000]
        self.n_shares = 0.010054921260545427
        self.file = file
        self.stop_loss_percentage =  0.01149031080475886
        self.take_profit_percentage =  0.054671359655351165
        self.file_mapping = {
            "APPLE": "data/aapl_5m_test.csv",
            "BITCOIN": "data/btc_project_test.csv"
        }
        self.load_data(self.file)
        self.indicators = {}
        self.active_indicators = []
        self.calculate_indicators() 
        self.define_buy_sell_signals() 
        self.run_signals()
        self.best_combination = None
        self.best_value = 100000
     
    
    def load_data(self, asset_type):
        file_name = self.file_mapping.get(asset_type)
        if not file_name:
            raise ValueError(f"Asset type '{asset_type}' not found in file mapping.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)
        

    def calculate_indicators(self):

        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=7)
        self.data['RSI'] = rsi_indicator.rsi()

        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
    
    
    def define_buy_sell_signals(self):
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal}
        }

    
    def activate_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)

    
    


    def rsi_buy_signal(self, row, prev_row=None):
        return row.RSI < 30

    
    def rsi_sell_signal(self, row, prev_row=None):
        return row.RSI > 70

    def calculate_max_drawdown(self):
        cumulative_returns = (1 + pd.Series(self.strategy_value).pct_change()).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = drawdown.max()  
        return max_drawdown
    

    
   

    
    def run_signals(self):
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data.apply(
                lambda row: self.indicators[indicator]['buy'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1
            )
            self.data[indicator + '_sell_signal'] = self.data.apply(
                lambda row: self.indicators[indicator]['sell'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1
            )

        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data[indicator + '_buy_signal'].astype(int)
            self.data[indicator + '_sell_signal'] = self.data[indicator + '_sell_signal'].astype(int)
    
    
    def execute_trades(self, best=False):
        
        if best:
            for indicator in self.best_combination:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.best_combination]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.best_combination]].sum(axis=1)
            total_active_indicators = len(self.best_combination)
        else:  
            for indicator in self.active_indicators:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.active_indicators]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.active_indicators]].sum(axis=1)
            total_active_indicators = len(self.active_indicators)

        
        for i, row in self.data.iterrows():

            
            if total_active_indicators <= 2:
                if self.data['total_buy_signals'].iloc[i] == total_active_indicators:
                    self._open_operation('long', row)
                elif self.data['total_sell_signals'].iloc[i] == total_active_indicators:
                    self._open_operation('short', row)

            
            else:
                if self.data['total_buy_signals'].iloc[i] > (total_active_indicators / 2):
                    self._open_operation('long', row)
                elif self.data['total_sell_signals'].iloc[i] > (total_active_indicators / 2):
                    self._open_operation('short', row)

            
            self.check_close_operations(row)

            
            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
            self.strategy_value.append(total_value)




    def _open_operation(self, operation_type, row):
        if operation_type == 'long':
            stop_loss = row['Close'] * (1 - self.stop_loss_percentage)  
            take_profit = row['Close'] * (1 + self.take_profit_percentage) 
        else:  
            stop_loss = row['Close'] * (1 + self.stop_loss_percentage)
            take_profit = row['Close'] * (1 - self.take_profit_percentage)

       
        self.operations.append(Operation(operation_type, row['Close'], row.name, self.n_shares, stop_loss, take_profit))

        
        if operation_type == 'long':
            self.cash -= row['Close'] * self.n_shares * (1 + self.com)
        else:  
            self.cash += row['Close'] * self.n_shares * (1 - self.com)  


    def check_close_operations(self, row):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= op.take_profit or row['Close'] <= op.stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= op.take_profit or row['Close'] >= op.stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * op.n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * op.n_shares * (1 + self.com)  
                   
                op.closed = True
                
    def calculate_operation_value(self, op, current_price):
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * op.n_shares if not op.closed else 0
        else:  
            return (op.bought_at - current_price) * op.n_shares if not op.closed else 0
        
    def run_best(self):  
        self.active_indicators = ['RSI'] # Sets the best combination of indicators manually (e.g., RSI here, but should be updated based on prior runs)
        print(f"Ejecutando con combinación de indicadores: {self.active_indicators}") # Print the combination being used for execution
        self.execute_trades() # Execute trades using the selected indicators
    
        max_drawdown = self.calculate_max_drawdown() # Calculate the maximum drawdown
    
        if max_drawdown < self.best_value: # If the current drawdown is lower than the best recorded drawdown  
            self.best_value = max_drawdown # Update the best value
            self.best_combination = self.active_indicators.copy() # Save the best combination of indicators

        print(f"Mejor combinación de indicadores: {self.best_combination} con un Max Drawdown de: {self.best_value}") # Print the final best combination of indicators and its drawdown

    def reset_strategy(self):
        self.operations.clear()
        self.cash = 1_000_000
        self.strategy_value = [1_000_000]      
        
    def scaled_returns(self):
        returns = self.data['Close'].pct_change()

        # Inicia el valor de la inversión con 1,000,000
        initial_investment = 1_000_000

        # Calcula el valor acumulado de la inversión
        self.data['Investment_Value'] = initial_investment * (1 + returns).cumprod()

        # Opcional: puedes querer guardar estos rendimientos escalados en el DataFrame
        scaled_returns = self.data['Investment_Value'] 

        return scaled_returns
    
    def passive_max_drawdown(self):
    
        cumulative_returns =  (1 + pd.Series(self.scaled_returns).pct_change()).cumprod()

        # Encuentra el máximo histórico de los valores acumulativos
        peak = cumulative_returns.cummax()

        # Calcula el drawdown como la diferencia relativa desde el pico
        drawdown = (peak - cumulative_returns) / peak

        # Encuentra y devuelve el máximo drawdown
        max_drawdown = drawdown.max()
        print(f"En estrategía pasiva el Max Drawdown es de: {max_drawdown}")
        return max_drawdown

    def plot_results(self): 
        self.scaled_returns()
        self.scaled_returns = self.scaled_returns()
        plt.figure(figsize=(12, 8))

  
        plt.plot(self.strategy_value, label='Strategy Value')
        plt.plot(self.scaled_returns,label='Passive Strategy Value') 

        plt.xlabel('Number of Trades or Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()




        

        
        