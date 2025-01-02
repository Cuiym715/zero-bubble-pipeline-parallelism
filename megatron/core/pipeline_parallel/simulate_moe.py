import random

def generate_sleep_times(file_path, num_steps, min_sleep=0, max_sleep=0.005):
    with open(file_path, 'w') as f:
        for _ in range(num_steps):
            sleep_time = random.uniform(min_sleep, max_sleep)
            f.write(f"{sleep_time}\n")

generate_sleep_times("sleep_times.txt", num_steps=10000)
