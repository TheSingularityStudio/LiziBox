
"""
配置管理模块 - 提供统一的配置加载和管理功能
"""
import os
import json
import threading
from typing import Dict, Any, Optional, Union
from .events import EventBus, Event, EventType
from .state import state_manager

class ConfigManager:
    """配置管理器"""
    _instance = None
    _lock = threading.Lock()  # 类级别锁，确保线程安全

    def __new__(cls):


            
        if cls._instance is None:


            
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance


    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config_file = 'config.json'
            self._settings: Dict[str, Any] = {}
            self._event_bus = EventBus()
            self._state_manager = state_manager
            self._initialized = True

            # 加载默认配置
            self._load_default_config()

            # 尝试从文件加载配置
            self.load_config()

    def _load_default_config(self) -> None:
        """加载默认配置"""
        self._settings = {
            # 窗口设置
            "window_title": "LiziEngine",
            "window_width": 800,
            "window_height": 600,

            # 网格设置
            "default_grid_width": 640,
            "default_grid_height": 480,
            "show_grid": True,
            "cell_size": 1,

            # 视图设置
            "cam_x": 0,
            "cam_y": 0,
            "cam_zoom": 1.0,

            # 向量场设置
            "default_brush_size": 1,
            "default_magnitude": 1.0,
            "vector_self_weight": 1.0,
            "vector_neighbor_weight": 0.1,
            "include_self": False,
            "enable_vector_average": False,
            "reverse_vector": False,

            # GPU计算设置
            "use_opencl_compute": True,
            "opencl_compute_threshold": 10000,  # 网格点数阈值，超过此值使用GPU计算
            "compute_shader_local_size_x": 16,
            "compute_shader_local_size_y": 16,
            "update_frequency": 30.0,

            # 渲染设置
            "vector_color": [0.2, 0.6, 1.0],
            "grid_color": [0.3, 0.3, 0.3],
            "background_color": [0.1, 0.1, 0.1],

            # 其他设置
            "enable_vector_smoothing": True,
            "vector_smooth_factor": 0.8,
            "enable_debug_output": False,
            "enable_event_output": True,
        }

    def load_config(self, config_file: Optional[str] = None) -> None:
        """从配置文件加载设置"""
        file_path = config_file or self._config_file
        if not os.path.exists(file_path):
            print(f"[配置管理] 配置文件 {file_path} 不存在，使用默认设置")
            return

        print(f"[配置管理] 正在加载配置文件: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    # JSON格式配置文件
                    file_settings = json.load(f)
                else:
                    # 旧的键值对格式配置文件
                    file_settings = {}
                    for line in f:
                        line = line.strip()
                        # 跳过空行和注释行
                        if not line or line.startswith('#'):
                            continue

                        # 解析键值对
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            file_settings[key] = self._parse_value(value)

            # 更新设置
            with self._lock:
                self._settings.update(file_settings)

            # 同步到状态管理器
            self._sync_to_state()

            # 发布配置加载事件
            self._event_bus.publish(Event(
                self._get_event_type("CONFIG_LOADED", EventType.VIEW_CHANGED),
                {"file_path": file_path},
                "ConfigManager"
            ))

            print(f"[配置管理] 配置加载完成，共加载 {len(file_settings)} 项设置")

        except Exception as e:
            print(f"[配置管理] 加载配置文件出错: {e}")

    def _parse_value(self, value: str) -> Union[str, int, float, bool, tuple]:
        """尝试将字符串值转换为适当的数据类型"""
        # 首先移除值后面的注释（如果有）
        if '#' in value:
            value = value.split('#')[0].strip()

        # 处理带引号的字符串
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]  # 移除引号

        # 尝试解析为布尔值
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False

        # 尝试解析为整数
        try:
            return int(value)
        except ValueError:
            pass

        # 尝试解析为浮点数
        try:
            return float(value)
        except ValueError:
            pass

        # 尝试解析为元组（用于RGB颜色等）
        if ',' in value:
            try:
                return tuple(float(x.strip()) for x in value.split(','))
            except ValueError:
                pass

        # 默认返回字符串
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，如果不存在则返回默认值"""
        with self._lock:
            return self._settings.get(key, default)

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """设置配置值"""
        with self._lock:
            old_value = self._settings.get(key)
            self._settings[key] = value

            # 同步到状态管理器
            self._state_manager.set(f"config_{key}", value)

            # 如果值发生变化，发布配置变更事件
            if old_value != value:
                self._event_bus.publish(Event(
                    self._get_event_type("CONFIG_CHANGED", EventType.VIEW_CHANGED),
                    {"key": key, "old_value": old_value, "new_value": value},
                    "ConfigManager"
                ))

        # 如果需要持久化，保存到文件
        if persist:
            self.save()

    def update(self, updates: Dict[str, Any], persist: bool = False) -> None:
        """批量更新配置"""
        changed = []

        with self._lock:
            for key, value in updates.items():
                old_value = self._settings.get(key)
                self._settings[key] = value

                # 同步到状态管理器
                self._state_manager.set(f"config_{key}", value)

                if old_value != value:
                    changed.append((key, old_value, value))

        # 如果有变化，发布批量配置变更事件
        if changed:
            self._event_bus.publish(Event(
                self._get_event_type("BATCH_CONFIG_CHANGED", EventType.VIEW_CHANGED),
                {"changes": changed},
                "ConfigManager"
            ))

        # 如果需要持久化，保存到文件
        if persist:
            self.save()

    def save(self, config_file: Optional[str] = None) -> None:
        """将当前配置保存到文件"""
        file_path = config_file or self._config_file

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    # JSON格式
                    json.dump(self._settings, f, indent=2, ensure_ascii=False)
                else:
                    # 旧的键值对格式
                    f.write("# LiziEngine 配置文件\n")
                    f.write("# 此文件记录了引擎的主要参数设置\n\n")

                    for key, value in sorted(self._settings.items()):
                        # 将值转换回字符串格式
                        if isinstance(value, bool):
                            value = 'true' if value else 'false'
                        elif isinstance(value, tuple):
                            value = ','.join(str(x) for x in value)
                        else:
                            value = str(value)

                        f.write(f"{key}={value}\n")

            print(f"[配置管理] 配置已保存到: {file_path}")

            # 发布配置保存事件
            self._event_bus.publish(Event(
                self._get_event_type("CONFIG_SAVED", EventType.VIEW_CHANGED),
                {"file_path": file_path},
                "ConfigManager"
            ))
        except Exception as e:
            print(f"[配置管理] 保存配置文件出错: {e}")
            
    def _get_event_type(self, type_name: str, default_type):
        """安全地获取事件类型"""
        try:
            from .events import EventType
            return getattr(EventType, type_name, default_type)
        except Exception:
            return default_type

    def _sync_to_state(self) -> None:
        """将配置同步到状态管理器"""
        # 直接设置状态值，避免触发事件和递归
        with self._state_manager._lock:
            for key, value in self._settings.items():
                self._state_manager._state[f"config_{key}"] = value

    def debug_print_all(self) -> None:
        """打印所有已加载的配置项，用于调试"""
        print("[配置管理] 所有配置项:")
        for key, value in sorted(self._settings.items()):
            print(f"  {key} = {value}")

# 确保类级别的锁已初始化
if not hasattr(ConfigManager, '_lock') or ConfigManager._lock is None:
    ConfigManager._lock = threading.Lock()

# 全局配置管理器实例
config_manager = ConfigManager()

# 便捷函数
def get_config(key: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return config_manager.get(key, default)

def set_config(key: str, value: Any, persist: bool = False) -> None:
    """设置配置值的便捷函数"""
    config_manager.set(key, value, persist)
