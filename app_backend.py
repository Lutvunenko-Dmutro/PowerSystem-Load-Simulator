import numpy as np
import random
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from PySide6.QtCore import QObject, Signal

from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class OptimizationBackend(QObject): 
    decomposition_result_ready = Signal(object, object, object, object) 
    evo_result_ready = Signal(object, float) 
    evo_status_update = Signal(str) 
    aco_result_ready = Signal(list, float, list) 
    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.time_buffer_size = 300
        self.known_period = 50 
        self.residuals_data = None
        
        # 7x7 матриця вартостей
        self.original_load_costs = np.full((7, 7), 999.0) # Створюємо 7x7
        edges = {
            (0, 1): 10.0, (0, 2): 10.0,
            (1, 3): 20.0, (1, 4): 30.0,
            (2, 4): 10.0, (2, 5): 20.0,
            (3, 4): 10.0,
            (4, 6): 15.0,
            (5, 6): 10.0
        }
        for (n1, n2), cost in edges.items():
            self.original_load_costs[n1, n2] = cost
            self.original_load_costs[n2, n1] = cost
        
        self.current_load_costs = self.original_load_costs.copy()
        self.broken_edges = set() 

    # --- 1. ДЕКОМПОЗИЦІЯ (Готує дані) ---
    def run_decomposition_analysis(self):
        self.log_message.emit("--- Запуск Декомпозиції (Statsmodels) ---")
        self.log_message.emit("...Аналіз часового ряду...")
        try:
            x_hist = np.arange(self.time_buffer_size)
            y_seasonal = 15 * np.sin(x_hist / self.known_period)
            y_noise = np.random.uniform(-3, 3, self.time_buffer_size)
            y_trend = 50 + x_hist * 0.01
            y_hist = y_trend + y_seasonal + y_noise
            for i in range(50, self.time_buffer_size, 70):
                y_hist[i:i+3] += 25 
            ts_data = pd.Series(
                y_hist, 
                index=pd.date_range(start='2025-01-01', periods=self.time_buffer_size, freq='h')
            )
            decomposition = seasonal_decompose(ts_data, model='additive', period=self.known_period)
            trend = decomposition.trend.bfill().ffill()
            seasonal = decomposition.seasonal
            resid = decomposition.resid.bfill().ffill()
            self.residuals_data = StandardScaler().fit_transform(resid.values.reshape(-1, 1))
            self.log_message.emit(f"Декомпозицію завершено. 'Піки' ізольовано.")
            self.decomposition_result_ready.emit(y_hist, trend.values, seasonal.values, resid.values)
        except Exception as e:
            self.log_message.emit(f"Statsmodels: Критична помилка: {e}")

    # --- 2. DE (НАВЧАЄ ДЕТЕКТОР АНОМАЛІЙ) ---
    def run_evolutionary_optimization(self):
        self.log_message.emit("--- Запуск DE (Навчання Детектора) ---")
        if self.residuals_data is None:
            self.log_message.emit("ПОМИЛКА: Спочатку запустіть '1. Аналіз Ряду'.")
            self.evo_status_update.emit("Помилка (Див. Журнал)")
            return
        self.evo_status_update.emit("Виконується...") 
        try:
            def model_fitness(params):
                contamination = params[0] 
                model = IsolationForest(contamination=contamination, random_state=42)
                labels = model.fit_predict(self.residuals_data)
                if np.sum(labels == -1) <= 1 or np.sum(labels == 1) <= 1:
                    return 1.0 
                score = silhouette_score(self.residuals_data, labels)
                return -score
            bounds = [(0.005, 0.1)] 
            result = differential_evolution(model_fitness, bounds, strategy='best1bin', maxiter=30, popsize=10)
            if result.success:
                best_contamination = result.x[0]
                self.log_message.emit(f"DE: Завершено. Знайдено contamination={best_contamination:.4f}")
                final_model = IsolationForest(contamination=best_contamination, random_state=42)
                final_labels = final_model.fit_predict(self.residuals_data)
                anomaly_indices = np.where(final_labels == -1)[0]
                self.evo_result_ready.emit(anomaly_indices, best_contamination)
            else:
                self.log_message.emit("DE: Оптимізація не вдалася (це дивно, перевірте логіку).")
                self.evo_status_update.emit("Не вдалося")
        except Exception as e:
            self.log_message.emit(f"DE (sklearn): Критична помилка: {e}")
            self.evo_status_update.emit("Помилка") 

    # --- 3. ACO (Пошук шляху) ---
    def run_aco_optimization(self):
        self.log_message.emit("--- Запуск ACO ---")
        self.log_message.emit("...Мурашині агенти шукають шлях...")
        
        try:
            load_costs = self.current_load_costs 
            num_nodes = 7      
            start_node = 0
            end_node = 6       
            num_ants = 10
            num_iterations = 20
            evaporation_rate = 0.5
            pheromone_alpha = 1.0
            heuristic_beta = 2.0
            pheromones = np.ones((num_nodes, num_nodes))
            heuristic = 1.0 / (load_costs + 1e-10) 
            best_path = None
            best_path_cost = float('inf')
            
            for iteration in range(num_iterations):
                all_paths = []
                for ant in range(num_ants):
                    path = [start_node]
                    current_node = start_node
                    visited = {start_node}
                    path_cost = 0.0
                    while current_node != end_node:
                        probabilities = []
                        possible_next_nodes = []
                        total_prob = 0.0
                        for next_node in range(num_nodes): 
                            if next_node not in visited and load_costs[current_node, next_node] != 999.0 and load_costs[current_node, next_node] != 9999.0:
                                tau = pheromones[current_node, next_node] ** pheromone_alpha
                                eta = heuristic[current_node, next_node] ** heuristic_beta
                                prob = tau * eta
                                probabilities.append(prob)
                                possible_next_nodes.append(next_node)
                                total_prob += prob
                        if total_prob == 0.0: break
                        probabilities = [p / total_prob for p in probabilities]
                        next_node = random.choices(possible_next_nodes, weights=probabilities, k=1)[0]
                        path.append(next_node)
                        visited.add(next_node)
                        path_cost += load_costs[current_node, next_node]
                        current_node = next_node
                    if current_node == end_node:
                        all_paths.append((path, path_cost))
                        if path_cost < best_path_cost:
                            best_path = path
                            best_path_cost = path_cost
                pheromones *= (1.0 - evaporation_rate)
                for path, cost in all_paths:
                    pheromone_deposit = 1.0 / cost 
                    for i in range(len(path) - 1):
                        pheromones[path[i], path[i+1]] += pheromone_deposit
                        pheromones[path[i+1], path[i]] += pheromone_deposit
            
            self.log_message.emit(f"ACO: Пошук завершено.")
            if best_path:
                self.aco_result_ready.emit(best_path, best_path_cost, list(self.broken_edges))
            else:
                self.log_message.emit("ACO: Не вдалося знайти шлях. Можливо, він заблокований.")
        
        except Exception as e:
            self.log_message.emit(f"ACO: Критична помилка: {e}")

    def toggle_edge_failure(self, edge_key):
        n1, n2 = edge_key
        if edge_key in self.broken_edges:
            self.broken_edges.remove(edge_key)
            original_cost = self.original_load_costs[n1, n2]
            self.current_load_costs[n1, n2] = original_cost
            self.current_load_costs[n2, n1] = original_cost
            self.log_message.emit(f"ЛЕП {edge_key} ВІДРЕМОНТОВАНО.")
        else:
            self.broken_edges.add(edge_key)
            self.current_load_costs[n1, n2] = 9999.0 
            self.current_load_costs[n2, n1] = 9999.0
            self.log_message.emit(f"🔥 АВАРІЯ! ЛЕП {edge_key} відключено.")
        
        self.run_aco_optimization()