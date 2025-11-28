
"""
状态管理模块 - 提供统一的应用状态管理
"""
import threading
from typing import Dict, Any, Callable, Optional, List
from .events import EventBus, Event, EventType

class StateManager:
    """线程安全的状态管理器"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._state: Dict[str, Any] = {}
            self._lock = threading.Lock()
            self._listeners: List[Callable[[str, Any, Any], None]] = []
            self._event_bus = EventBus()
            self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value: Any, notify: bool = True) -> None:
        """设置状态值"""
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value

            # 通知监听器状态已改变
            if notify and old_value != value:
                self._notify_listeners(key, old_value, value)

                # 发布状态变更事件
                self._event_bus.publish(Event(
                    self._get_event_type("STATE_CHANGED", EventType.VIEW_CHANGED),
                    {"key": key, "old_value": old_value, "new_value": value},
                    "StateManager"
                ))

    def update(self, updates: Dict[str, Any], notify: bool = True) -> None:
        """批量更新状态"""
        with self._lock:
            changed_items = []

            for key, value in updates.items():
                old_value = self._state.get(key)
                if old_value != value:
                    self._state[key] = value
                    changed_items.append((key, old_value, value))

            # 通知监听器状态已改变
            if notify and changed_items:
                for key, old_value, new_value in changed_items:
                    self._notify_listeners(key, old_value, new_value)

                # 批量状态变更事件
                self._event_bus.publish(Event(
                    EventType.BATCH_STATE_CHANGED if hasattr(EventType, 'BATCH_STATE_CHANGED') else EventType.VIEW_CHANGED,
                    {"changes": changed_items},
                    "StateManager"
                ))

    def add_listener(self, listener: Callable[[str, Any, Any], None]) -> None:
        """添加状态变化监听器"""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[str, Any, Any], None]) -> None:
        """移除状态变化监听器"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def clear_listeners(self) -> None:
        """清空所有监听器"""
        with self._lock:
            self._listeners.clear()

    def _notify_listeners(self, key: str, old_value: Any, new_value: Any) -> None:
        """通知所有监听器状态已改变"""
        for listener in self._listeners:
            try:
                listener(key, old_value, new_value)
            except Exception as e:
                print(f"[状态管理] 通知监听器时出错: {e}")

    def get_all(self) -> Dict[str, Any]:
        """获取所有状态的副本"""
        with self._lock:
            return self._state.copy()

    def reset(self, keys: Optional[List[str]] = None) -> None:
        """重置指定键的状态或全部状态"""
        with self._lock:
            if keys is None:
                # 重置所有状态
                self._state.clear()
            else:
                # 只重置指定的键
                for key in keys:
                    if key in self._state:
                        del self._state[key]

            # 发布状态重置事件
            self._event_bus.publish(Event(
                self._get_event_type("STATE_RESET", EventType.VIEW_CHANGED),
                {"keys": keys},
                "StateManager"
            ))

    def _get_event_type(self, type_name: str, default_type):
        """安全地获取事件类型"""
        try:
            from .events import EventType
            return getattr(EventType, type_name, default_type)
        except Exception:
            return default_type

# 全局状态管理器实例
state_manager = StateManager()

# 便捷函数
def get_state(key: str, default: Any = None) -> Any:
    """获取状态值的便捷函数"""
    return state_manager.get(key, default)

def set_state(key: str, value: Any, notify: bool = True) -> None:
    """设置状态值的便捷函数"""
    state_manager.set(key, value, notify)

def update_state(updates: Dict[str, Any], notify: bool = True) -> None:
    """批量更新状态的便捷函数"""
    state_manager.update(updates, notify)
