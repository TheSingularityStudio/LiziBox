import os

class Config:
    """配置管理类，负责读取和应用config.txt中的设置"""

    def __init__(self, config_file='config.txt'):
        """初始化配置管理器，从指定文件加载配置"""
        self.config_file = config_file
        self.settings = {}
        self.load_config()

    def load_config(self):
        """从配置文件加载设置"""
        if not os.path.exists(self.config_file):
            print(f"配置文件 {self.config_file} 不存在，使用默认设置")
            return
        
        print(f"正在加载配置文件: {self.config_file}")

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
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

                        # 尝试转换为适当的数据类型
                        self.settings[key] = self._parse_value(value)
                        print(f"加载配置项: {key} = {self.settings[key]}")
        except Exception as e:
            print(f"加载配置文件出错: {e}")

    def _parse_value(self, value):
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

    def get(self, key, default=None):
        """获取配置值，如果不存在则返回默认值"""
        value = self.settings.get(key, default)
        #print(f"获取配置项: {key} = {value}")
        return value
    
    def debug_print_all(self):
        """打印所有已加载的配置项，用于调试"""
        print("所有配置项:")
        for key, value in self.settings.items():
            print(f"  {key} = {value}")

    def set(self, key, value):
        """设置配置值"""
        self.settings[key] = value

    def save(self):
        """将当前配置保存到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write("# LiziEngine 配置文件")
                f.write("# 此文件记录了引擎的主要参数设置")

                for key, value in sorted(self.settings.items()):
                    # 将值转换回字符串格式
                    if isinstance(value, bool):
                        value = 'true' if value else 'false'
                    elif isinstance(value, tuple):
                        value = ','.join(str(x) for x in value)
                    else:
                        value = str(value)

                    f.write(f"{key}={value}")
        except Exception as e:
            print(f"保存配置文件出错: {e}")

# 创建全局配置实例
config = Config()

# 测试配置是否正确加载
if __name__ == "__main__":
    print("测试配置文件加载...")
    config.debug_print_all()
