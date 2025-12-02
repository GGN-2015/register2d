import time

# 将 BAN_ALL 设置为 True 可以禁用全部计时器
BAN_ALL = False
__global_obj_timer_name_dict = {}

# 禁用所有的计时器输出
def ban_all_timer():
    global BAN_ALL
    BAN_ALL = True

# 在禁用所有计时器输出后，重新允许所有计时器输出
def allow_all_timer():
    global BAN_ALL
    BAN_ALL = False

def begin_timer(name:str):
    assert __global_obj_timer_name_dict.get(name) is None
    __global_obj_timer_name_dict[name] = time.time()

def end_timer(name:str, disp:bool=True) -> float:
    assert __global_obj_timer_name_dict.get(name) is not None
    
    # 计算花费的时间
    time_cost = time.time() - __global_obj_timer_name_dict[name]

    # 删除计时器
    del __global_obj_timer_name_dict[name]

    # 显示计时器
    if disp and not BAN_ALL:

        # 美元号开头的用黄色显示
        if name.startswith("$"):
            print(f"Timer [\033[1;33m{name:30s}\033[0m]: {time_cost:13.6f}s")

        # 其他的用绿色显示
        else:
            print(f"Timer [\033[1;32m{name:30s}\033[0m]: {time_cost:13.6f}s")
    return time_cost
