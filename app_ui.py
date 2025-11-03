import sys
import numpy as np
import random
import pyqtgraph as pg
import inspect 
import textwrap 

from radon.visitors import ComplexityVisitor 
from radon.raw import analyze as analyze_raw 

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QGridLayout, QPushButton, QTextEdit, 
    QTabWidget, QLabel, QGroupBox, QFrame
)
from PySide6.QtCore import QTimer, Qt, Signal, QObject 
from app_backend import OptimizationBackend
from help_content import HTML_THEORY_CONTENT


class PowerSystemDashboard(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(f"Симулятор моніторингу (Д. Литвиненко) - v3.5 (Аналіз коду)")
        self.setGeometry(100, 100, 1200, 800) 

        main_layout = QGridLayout()
        
        # --- 1. Ліва панель: Графіки ---
        left_panel_layout = QVBoxLayout()
        self.main_tabs = QTabWidget() 
        self.main_tabs.addTab(self.create_live_monitor_tab(), "📈 Моніторинг (Live)")
        self.decomposition_tab = self.create_decomposition_tab() 
        self.main_tabs.addTab(self.decomposition_tab, "📊 Декомпозиція (Аналіз)")
        self.main_tabs.addTab(self.create_network_map_tab(), "🗺️ Карта Мережі") 
        left_panel_layout.addWidget(self.main_tabs)
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel_layout)
        main_layout.addWidget(left_panel_widget, 0, 0) 
        
        # --- 2. Права панель: Управління, Довідка, Журнал, Код ---
        right_panel_layout = QVBoxLayout()
        self.control_tabs = QTabWidget()
        self.control_tabs.addTab(self.create_optimization_tab(), "⚙️ Оптимізація та Стан")
        self.control_tabs.addTab(self.create_theory_tab(), "📚 Довідка (по темі)")
        self.control_tabs.addTab(self.create_code_report_tab(), "🔍 Код (Звіт/Аналіз)")
        self.control_tabs.addTab(self.create_log_tab(), "Console Журнал")
        right_panel_layout.addWidget(self.control_tabs)
        
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel_layout)
        main_layout.addWidget(right_panel_widget, 0, 1)
        
        main_layout.setColumnStretch(0, 7)
        main_layout.setColumnStretch(1, 3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.backend = OptimizationBackend()
        self.connect_signals()

        self.timer = QTimer()
        self.timer.setInterval(50) 
        self.timer.timeout.connect(self.update_live_plot)
        self.timer.start()
        
        self.current_optimal_path = []
        
        self.log_to_gui("Дашборд запущено (v3.5 - Аналіз коду).")
        self.on_main_tab_changed(0) 

    def log_to_gui(self, message):
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )

    def update_live_plot(self):
        self.data_buffer = np.roll(self.data_buffer, -1)
        base_load = 50 + np.sin(len(self.data_buffer) / 50.0) * 15
        noise = random.uniform(-3, 3)
        new_data_point = base_load + noise
        if random.random() < 0.015: 
            new_data_point += 30 
        self.data_buffer[-1] = new_data_point
        self.live_data_line.setData(self.data_buffer)
        if new_data_point > 80:
            self.live_data_line.setPen(pg.mkPen(color=(255, 0, 0), width=3)) 
        else:
            self.live_data_line.setPen(pg.mkPen(color=(0, 0, 255), width=2)) 

    def update_decomposition_plots(self, orig, trend, seasonal, resid):
        self.anomaly_markers.clear()
        self.decomp_orig_line.setData(orig)
        self.decomp_trend_line.setData(trend)
        self.decomp_seasonal_line.setData(seasonal)
        self.decomp_resid_line.setData(resid)
        self.decomp_resid_plot.setYRange(min(resid) - 5, max(resid) + 5)
        self.log_to_gui("Візуалізація декомпозиції завершена.")
        self.main_tabs.setCurrentWidget(self.decomposition_tab)
        self.forecast_status.setText("Аналіз готовий")

    def update_anomaly_detector(self, anomaly_indices, contamination_val):
        self.de_anomaly_count_val.setText(f"{len(anomaly_indices)} шт.")
        self.de_contamination_val.setText(f"{contamination_val*100:.2f} %")
        self.de_status.setText("Готово")
        self.visualize_anomalies(anomaly_indices)

    def update_de_status(self, status):
        self.de_status.setText(status)

    def update_aco_results(self, path, cost, broken_edges):
        path_str = " -> ".join(map(str, path)) if path else "N/A"
        cost_str = f"{cost:.2f}" if path else "N/A"
        
        self.aco_path_val.setText(path_str)
        self.aco_cost_val.setText(cost_str)
        self.aco_status.setText("Готово")
        
        self.current_optimal_path = path if path else []
        
        self.visualize_network_state(path, broken_edges) 

    def visualize_anomalies(self, anomaly_indices):
        self.log_to_gui(f"Візуалізація {len(anomaly_indices)} аномалій...")
        self.anomaly_markers.clear()
        all_x_data = self.decomp_resid_line.xData
        all_y_data = self.decomp_resid_line.yData
        if all_x_data is None or all_y_data is None: return
        anomaly_x = all_x_data[anomaly_indices]
        anomaly_y = all_y_data[anomaly_indices]
        self.anomaly_markers.setData(x=anomaly_x, y=anomaly_y)
        self.main_tabs.setCurrentWidget(self.decomposition_tab)
        
    def on_map_clicked(self, event):
        if not event.button() == Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        view_pos = self.network_plot_widget.getViewBox().mapSceneToView(pos)
        click_x, click_y = view_pos.x(), view_pos.y()
        min_dist = float('inf')
        clicked_edge_key = None
        for key, edge_item in self.edge_items.items():
            x1, x2 = edge_item.xData
            y1, y2 = edge_item.yData
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0: 
                dist = np.hypot(click_x - x1, click_y - y1)
            else:
                t = ((click_x - x1) * dx + (click_y - y1) * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t)) 
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.hypot(click_x - proj_x, click_y - proj_y)
            if dist < min_dist:
                min_dist = dist
                clicked_edge_key = key
        click_threshold = 1.0 
        if clicked_edge_key and min_dist < click_threshold:
            self.log_to_gui(f"Клік на ЛЕП {clicked_edge_key}. Імітація аварії/ремонту...")
            self.backend.toggle_edge_failure(clicked_edge_key)
        else:
            self.log_to_gui("Клік повз ЛЕП (або недостатньо близько).")

        
    def visualize_network_state(self, optimal_path, broken_edges):
        self.log_to_gui(f"Оновлення карти. Шлях: {optimal_path}, Аварії: {broken_edges}")
        default_pen = pg.mkPen(color=(150, 150, 150), width=2, style=Qt.DotLine)
        optimal_pen = pg.mkPen(color=(0, 200, 0), width=6) 
        broken_pen = pg.mkPen(color=(255, 0, 0), width=5, style=Qt.DashLine)
        optimal_set = set()
        if optimal_path:
            for i in range(len(optimal_path) - 1):
                optimal_set.add(tuple(sorted((optimal_path[i], optimal_path[i+1]))))
        broken_set = set(broken_edges) 
        for key, edge_item in self.edge_items.items():
            if key in broken_set:
                edge_item.setPen(broken_pen) 
            elif key in optimal_set:
                edge_item.setPen(optimal_pen)
            else:
                edge_item.setPen(default_pen)


    def on_main_tab_changed(self, index):
        """
        Функція: бере код "наживо" з файлу,
        аналізує його "якість" (метрики) і показує у вкладці "Код (Звіт)".
        """
        if not hasattr(self, 'code_report_display'):
            return 
        self.code_report_display.clear()
        
        source_code = ""
        title = ""
        target_function = None
        
        try:
            if index == 0:
                title = "--- Код для 'Моніторинг' ---"
                target_function = self.create_live_monitor_tab
            elif index == 1:
                title = "--- Код для 'Декомпозиція' ---"
                target_function = self.create_decomposition_tab
            elif index == 2:
                title = "--- Код для 'Карта Мережі' ---"
                target_function = self.create_network_map_tab
            
            if target_function:
                # 1. Отримуємо код
                source_code = inspect.getsource(target_function)
                cleaned_code = textwrap.dedent(source_code) 
                
                # 2. Аналіз
                raw_analysis = analyze_raw(cleaned_code)
                lines_of_code = raw_analysis.loc 
                
                visitor = ComplexityVisitor.from_code(cleaned_code)
                complexity = 0
                if visitor.functions:
                    complexity = visitor.functions[0].complexity 
                
                quality = "Добре (Легко тестувати)"
                if complexity > 7:
                    quality = "Середнє (Варто спростити)"
                if complexity > 12:
                    quality = "Погано (Дуже заплутано)"

                # 3. Формуємо звіт
                report = (
                    f"--- Статичний Аналіз (для Звіту) ---\n\n"
                    f"Функція:\t{target_function.__name__}\n"
                    f"Рядки Коду (LOC):\t{lines_of_code}\n"
                    f"Цикломатична Складність:\t{complexity} ({quality})\n\n"
                    f"--- Вихідний Код ---\n\n"
                    f"{cleaned_code}"
                )
                self.code_report_display.setText(report)
            
        except Exception as e:
            self.code_report_display.setText(f"Помилка інспектування коду: {e}")

    # --- Функції налаштування GUI ---
        
    def connect_signals(self):
        self.btn_decomp.clicked.connect(self.backend.run_decomposition_analysis)
        self.btn_evo.clicked.connect(self.backend.run_evolutionary_optimization)
        self.btn_aco.clicked.connect(self.backend.run_aco_optimization)
        self.backend.log_message.connect(self.log_to_gui)
        self.backend.decomposition_result_ready.connect(self.update_decomposition_plots)
        self.backend.evo_result_ready.connect(self.update_anomaly_detector)
        self.backend.evo_status_update.connect(self.update_de_status)  
        self.backend.aco_result_ready.connect(self.update_aco_results)
        self.network_plot_widget.scene().sigMouseClicked.connect(self.on_map_clicked)
        self.main_tabs.currentChanged.connect(self.on_main_tab_changed)
        
    def create_live_monitor_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout()
        self.live_plot_widget = pg.PlotWidget()
        self.live_plot_widget.setBackground('w')
        self.live_plot_widget.setTitle("Моніторинг навантаження (Live)", color='k', size='14pt')
        self.live_plot_widget.setLabel('left', 'Навантаження (МВт)', color='k')
        self.live_plot_widget.setLabel('bottom', 'Час (ticks)', color='k')
        self.live_plot_widget.showGrid(x=True, y=True)
        self.live_data_line = self.live_plot_widget.plot(pen=pg.mkPen(color=(0, 0, 255), width=2))
        limit_line = pg.InfiniteLine(pos=80, angle=0, movable=False, pen=pg.mkPen('r', width=2, style=Qt.DashLine))
        self.live_plot_widget.addItem(limit_line)
        self.live_plot_widget.setYRange(0, 100)
        self.time_buffer_size = 300 
        self.data_buffer = np.zeros(self.time_buffer_size)
        layout.addWidget(self.live_plot_widget)
        tab_widget.setLayout(layout)
        return tab_widget

    def create_decomposition_tab(self):
        tab_widget = QWidget()
        layout = QGridLayout() 
        self.decomp_orig_plot = pg.PlotWidget(title="1. Original (Вхідний сигнал)")
        self.decomp_trend_plot = pg.PlotWidget(title="2. Trend (Загальний тренд)")
        self.decomp_seasonal_plot = pg.PlotWidget(title="3. Seasonal (Сезонність/Цикли)")
        self.decomp_resid_plot = pg.PlotWidget(title="4. Residual (🔥 Залишки / ПІКИ)")
        plots = [self.decomp_orig_plot, self.decomp_trend_plot, self.decomp_seasonal_plot, self.decomp_resid_plot]
        for plot in plots:
            plot.setBackground('w')
            plot.showGrid(x=True, y=True)
            plot.getPlotItem().setLabel('left', 'Навант. (МВт)')
        layout.addWidget(self.decomp_orig_plot, 0, 0)
        layout.addWidget(self.decomp_trend_plot, 0, 1)
        layout.addWidget(self.decomp_seasonal_plot, 1, 0)
        layout.addWidget(self.decomp_resid_plot, 1, 1)
        self.decomp_orig_line = self.decomp_orig_plot.plot(pen='b')
        self.decomp_trend_line = self.decomp_trend_plot.plot(pen='g')
        self.decomp_seasonal_line = self.decomp_seasonal_plot.plot(pen='c')
        self.decomp_resid_line = self.decomp_resid_plot.plot(pen='k')
        self.anomaly_markers = pg.ScatterPlotItem(size=15, pen=pg.mkPen('r', width=3), brush=pg.mkBrush(255, 0, 0, 0), symbol='o')
        self.decomp_resid_plot.addItem(self.anomaly_markers)
        tab_widget.setLayout(layout)
        return tab_widget

    def create_network_map_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout()
        self.network_plot_widget = pg.PlotWidget()
        self.network_plot_widget.setBackground('w')
        self.network_plot_widget.setTitle("Карта Енергомережі (7 Вузлів, Cіль - Вузол 6)", color='k', size='14pt')
        self.network_plot_widget.getPlotItem().hideAxis('left')
        self.network_plot_widget.getPlotItem().hideAxis('bottom')
        self.network_plot_widget.setAspectLocked(True)
        pos = np.array([ 
            [0, 10],   # Node 0 (Старт)
            [5, 5],    # Node 1
            [5, 15],   # Node 2
            [10, 0],   # Node 3
            [15, 10],  # Node 4
            [10, 20],  # Node 5
            [20, 10]   # Node 6 (Ціль)
        ])
        adj = np.array([ 
            [0, 1], [0, 2], [1, 3], [1, 4], [2, 4], [2, 5],
            [3, 4], [4, 6], [5, 6] 
        ])
        self.edge_items = {} 
        default_pen = pg.mkPen(color=(150, 150, 150), width=2, style=Qt.DotLine)
        for n1, n2 in adj:
            x_coords = [pos[n1, 0], pos[n2, 0]]
            y_coords = [pos[n1, 1], pos[n2, 1]]
            key = tuple(sorted((n1, n2)))
            edge_item = pg.PlotCurveItem(
                x=x_coords, y=y_coords, pen=default_pen, skipFiniteCheck=True
            )
            self.network_plot_widget.addItem(edge_item)
            self.edge_items[key] = edge_item 
        non_target_nodes_pos = np.array([pos[i] for i in range(len(pos)) if i != 6])
        nodes = pg.ScatterPlotItem(
            pos=non_target_nodes_pos, size=15, pen=pg.mkPen('k'), 
            brush=pg.mkBrush('c'), hoverable=True, 
            hoverPen=pg.mkPen('r', width=2)
        )
        self.network_plot_widget.addItem(nodes)
        target_node_pos = np.array([pos[6]])
        target_node = pg.ScatterPlotItem(
            pos=target_node_pos, size=20, pen=pg.mkPen('k', width=2), 
            brush=pg.mkBrush('orange'), symbol='star', 
            hoverable=True, hoverPen=pg.mkPen('purple', width=3)
        )
        self.network_plot_widget.addItem(target_node)
        for i, p in enumerate(pos):
            text_item = pg.TextItem(f"{i}", anchor=(0.5, 0.5))
            text_item.setPos(p[0], p[1] + 1.5) 
            self.network_plot_widget.addItem(text_item)
        layout.addWidget(self.network_plot_widget)
        tab_widget.setLayout(layout)
        return tab_widget

    def create_optimization_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout()
        button_box = QGroupBox("Панель Управління")
        button_layout = QVBoxLayout()
        self.btn_decomp = QPushButton("1. Аналіз Ряду (Statsmodels)")
        self.btn_evo = QPushButton("2. DE (Навчити Детектор Аномалій)")
        self.btn_aco = QPushButton("3. ACO (Пошук шляху на карті)")
        button_layout.addWidget(self.btn_decomp) 
        button_layout.addWidget(self.btn_evo)
        button_layout.addWidget(self.btn_aco)
        button_box.setLayout(button_layout)
        layout.addWidget(button_box)
        status_box = QGroupBox("Панель Стану")
        status_layout = QGridLayout()
        status_layout.addWidget(QLabel("<b>Аналіз Часового Ряду:</b>"), 0, 0, 1, 2)
        status_layout.addWidget(QLabel("Статус:"), 1, 0)
        self.forecast_status = QLabel("Очікування...")
        status_layout.addWidget(self.forecast_status, 1, 1)
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine); line1.setFrameShadow(QFrame.Sunken)
        status_layout.addWidget(line1, 2, 0, 1, 2)
        status_layout.addWidget(QLabel("<b>Детектор Аномалій (DE):</b>"), 3, 0, 1, 2)
        status_layout.addWidget(QLabel("Знайдено аномалій:"), 4, 0)
        self.de_anomaly_count_val = QLabel("N/A")
        status_layout.addWidget(self.de_anomaly_count_val, 4, 1)
        status_layout.addWidget(QLabel("Оптим. 'Contamination':"), 5, 0)
        self.de_contamination_val = QLabel("N/A")
        status_layout.addWidget(self.de_contamination_val, 5, 1)
        status_layout.addWidget(QLabel("Статус:"), 6, 0)
        self.de_status = QLabel("Очікування...") 
        status_layout.addWidget(self.de_status, 6, 1)
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine); line2.setFrameShadow(QFrame.Sunken)
        status_layout.addWidget(line2, 7, 0, 1, 2)
        status_layout.addWidget(QLabel("<b>Статус Мережі (ACO):</b>"), 8, 0, 1, 2)
        status_layout.addWidget(QLabel("Найкращий шлях:"), 9, 0)
        self.aco_path_val = QLabel("N/A")
        status_layout.addWidget(self.aco_path_val, 9, 1)
        status_layout.addWidget(QLabel("Вартість (навант.):"), 10, 0)
        self.aco_cost_val = QLabel("N/A")
        status_layout.addWidget(self.aco_cost_val, 10, 1)
        status_layout.addWidget(QLabel("Статус:"), 11, 0)
        self.aco_status = QLabel("Очікування...")
        status_layout.addWidget(self.aco_status, 11, 1)
        status_box.setLayout(status_layout)
        layout.addWidget(status_box)
        layout.addStretch() 
        tab_widget.setLayout(layout)
        return tab_widget

    def create_theory_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout()
        theory_text = QTextEdit()
        theory_text.setReadOnly(True)
        
        # Використовуємо імпортований HTML
        theory_text.setHtml(HTML_THEORY_CONTENT) 
        
        layout.addWidget(theory_text)
        tab_widget.setLayout(layout)
        return tab_widget

    def create_code_report_tab(self):
        """Створює вкладку 'Код' з темним стилем, як у VS."""
        tab_widget = QWidget()
        layout = QVBoxLayout()
        
        label = QLabel("Код UI для поточної вкладки (для звіту):")
        layout.addWidget(label)
        
        self.code_report_display = QTextEdit()
        self.code_report_display.setReadOnly(True)
        self.code_report_display.setFontFamily("Consolas") 
        
        self.code_report_display.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E; 
                color: #D4D4D4; 
                font-size: 10pt;
                padding: 5px;
            }
        """)
        
        layout.addWidget(self.code_report_display)
        
        tab_widget.setLayout(layout)
        return tab_widget

    def create_log_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout()
        log_label = QLabel("Тут з'являтимуться статусні повідомлення та помилки:")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setFontFamily("Consolas")
        layout.addWidget(self.log_text_edit)
        tab_widget.setLayout(layout)
        return tab_widget