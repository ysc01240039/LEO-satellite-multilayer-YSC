import numpy as np
import json
from satellite_motion import SatelliteMotionModel, BoundaryHandler

class SimulationFramework:
    def __init__(self, config):
        """
        仿真框架
        
        参数:
        - config: 配置字典
        """
        self.config = config
        self.n_satellites = config.get('n_satellites', 1000)
        self.box_size = config.get('box_size', 30)  # 扩展模拟空间到30
        self.dt = config.get('dt', 0.1)
        self.time_steps = config.get('time_steps', 1000)
        
        # 初始化卫星
        self.satellites = []
        self.initialize_satellites()
        
        # 初始化运动模型和边界处理
        self.motion_model = SatelliteMotionModel(
            motion_type=config.get('motion_type', 'hybrid'),
            noise_std=config.get('noise_std', 0.5),
            orbit_radius=config.get('orbit_radius', 7000),
            angular_velocity=config.get('angular_velocity', 7.292e-5)
        )
        
        self.boundary_handler = BoundaryHandler(
            boundary_type=config.get('boundary_type', 'periodic'),
            box_size=self.box_size
        )
        
        # 记录结果
        self.results = {
            'positions': [],
            'times': [],
            'cores': []
        }
        
        # 时间计数器
        self.time = 0
    
    def initialize_satellites(self):
        """
        初始化卫星位置
        """
        for i in range(self.n_satellites):
            # 随机初始化位置
            position = np.random.uniform(-self.box_size/2, self.box_size/2, 3)
            self.satellites.append({
                'id': i,
                'position': position,
                'velocity': np.zeros(3),
                'load': 0.0,
                'queue_length': 0,
                'is_core': False,
                'is_core_head': False
            })
    
    def update_satellites(self):
        """
        更新所有卫星的位置
        """
        for satellite in self.satellites:
            # 更新位置
            new_position = self.motion_model.update_position(satellite['position'], self.dt)
            # 处理边界条件
            new_position = self.boundary_handler.handle_boundary(new_position)
            # 更新速度（简化）
            satellite['velocity'] = (new_position - satellite['position']) / self.dt
            # 更新位置
            satellite['position'] = new_position
    
    def detect_cores(self):
        """
        检测核结构
        """
        # 重置核状态
        for satellite in self.satellites:
            satellite['is_core'] = False
            satellite['is_core_head'] = False
        
        # 基于局部负载检测核
        for i, satellite in enumerate(self.satellites):
            # 计算邻居负载
            neighbor_loads = []
            for j, other in enumerate(self.satellites):
                if i != j:
                    distance = np.linalg.norm(satellite['position'] - other['position'])
                    if distance < 5.0:  # 5.0的邻居半径
                        neighbor_loads.append(other['load'])
            
            # 检测核候选
            if neighbor_loads and satellite['load'] > max(neighbor_loads) * 1.1:  # 调整阈值到1.1
                satellite['is_core'] = True
        
        # 选举核头
        core_candidates = [s for s in self.satellites if s['is_core']]
        if core_candidates:
            # 基于负载和队列长度选举核头
            for core in core_candidates:
                core['strength'] = core['load'] + 0.15 * (core['queue_length'] ** 2)  # 调整系数
            
            # 每个核集群选举一个核头
            # 简化处理：选择强度最高的
            core_head = max(core_candidates, key=lambda x: x['strength'])
            core_head['is_core_head'] = True
    
    def update_load(self):
        """
        更新卫星负载
        """
        for satellite in self.satellites:
            # 模拟时间相关的负载变化（更符合实际网络流量）
            time_factor = np.sin(self.time * 0.1) * 0.2  # 时间周期性变化
            random_factor = np.random.normal(0, 0.1)  # 随机波动
            
            # 考虑位置相关的负载（模拟地面需求分布）
            position_factor = np.exp(-np.linalg.norm(satellite['position']) / 10) * 0.3
            
            # 计算负载变化
            load_change = time_factor + random_factor + position_factor
            satellite['load'] = max(0, satellite['load'] + load_change)
            
            # 模拟队列长度（考虑负载的非线性增长）
            satellite['queue_length'] = int(satellite['load'] * 10 + satellite['load'] ** 2 * 5)
    
    def run_simulation(self):
        """
        运行仿真
        """
        print(f"开始仿真，卫星数: {self.n_satellites}, 时间步: {self.time_steps}")
        
        for step in range(self.time_steps):
            if step % 100 == 0:
                print(f"时间步: {step}/{self.time_steps}")
            
            # 更新时间
            self.time = step * self.dt
            
            # 更新负载
            self.update_load()
            # 更新卫星位置
            self.update_satellites()
            # 检测核
            self.detect_cores()
            
            # 记录结果
            if step % 10 == 0:  # 每10步记录一次
                positions = [s['position'].tolist() for s in self.satellites]
                cores = [s['id'] for s in self.satellites if s['is_core']]
                core_heads = [s['id'] for s in self.satellites if s['is_core_head']]
                
                self.results['positions'].append(positions)
                self.results['times'].append(step * self.dt)
                self.results['cores'].append({
                    'core_ids': cores,
                    'core_head_ids': core_heads
                })
        
        print("仿真完成")
        
    def save_results(self, filename):
        """
        保存仿真结果
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果保存到: {filename}")

# 测试代码
if __name__ == "__main__":
    # 配置
    config = {
        'n_satellites': 100,
        'box_size': 30,  # 扩展模拟空间
        'dt': 0.1,
        'time_steps': 500,
        'motion_type': 'hybrid',
        'noise_std': 0.5,
        'boundary_type': 'periodic'
    }
    
    # 初始化仿真框架
    sim = SimulationFramework(config)
    # 运行仿真
    sim.run_simulation()
    # 保存结果
    sim.save_results('simulation_results.json')