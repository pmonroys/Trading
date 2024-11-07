# Costco Trading Simulation with GAN-Generated Scenarios

## Project Overview
The Costco Trading Simulation project is a comprehensive initiative aimed at developing, backtesting, and analyzing trading strategies using Generative Adversarial Networks (GANs) to simulate multiple market scenarios. This project leverages quantitative methods and algorithmic trading strategies to provide valuable insights for traders and investors interested in assessing strategy performance under various stop-loss and take-profit levels.

The project focuses on Python as the primary programming language, utilizing essential libraries to implement GANs for data generation and perform backtesting. Rather than directly optimizing hyperparameters, the project tests various levels of stop-loss and take-profit to identify robust performance patterns across simulated and historical data.

This project centers on the Costco (COST) stock, analyzing market trends and volatility through GAN-generated price scenarios. By systematically applying algorithmic trading strategies in these scenarios, the project aims to refine buy and sell signals and inform more resilient trading decisions.

## Project Structure
The project is organized to facilitate clarity, reproducibility, and collaborative development. Key components include:

- `data/`: Contains 10 years of historical Costco data, used to train the GAN and perform backtesting.
- `models/`: Stores the trained GAN models used to generate multiple hypothetical market scenarios.
- `strategies/`: Code implementing the algorithmic trading strategy and the benchmark passive strategy.
- `notebooks/`: Jupyter notebook for visualizing results, including charts, conclusions, and performance metrics.
- `requirements.txt`: Specifies required libraries and their versions.

## Installation and Setup
To set up the project environment, first ensure Python is installed, then install dependencies:

```bash
pip install -r requirements.txt


# Authors

* Castelan Bryan Edwin
* Monroy Salcido Paulina Monroy
* Leon Ortiz Ulises Rodrigo
