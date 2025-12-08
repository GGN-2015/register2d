import time

# Setting BAN_ALL to True disables all timers
BAN_ALL = False
__global_obj_timer_name_dict = {}

# Disable output from all timers
def ban_all_timer():
    global BAN_ALL
    BAN_ALL = True

# Re-enable output from all timers after they have been disabled
def allow_all_timer():
    global BAN_ALL
    BAN_ALL = False

def begin_timer(name: str):
    assert __global_obj_timer_name_dict.get(name) is None
    __global_obj_timer_name_dict[name] = time.time()

def end_timer(name: str, disp: bool = True) -> float:
    assert __global_obj_timer_name_dict.get(name) is not None
    
    # Calculate elapsed time
    time_cost = time.time() - __global_obj_timer_name_dict[name]

    # Remove the timer from the dictionary
    del __global_obj_timer_name_dict[name]

    # Display the timer result
    if disp and not BAN_ALL:

        # Names starting with $ are displayed in yellow
        if name.startswith("$"):
            print(f"Timer [\033[1;33m{name:35s}\033[0m]: {time_cost:13.6f}s")

        # Other names are displayed in green
        else:
            print(f"Timer [\033[1;32m{name:35s}\033[0m]: {time_cost:13.6f}s")
    return time_cost
