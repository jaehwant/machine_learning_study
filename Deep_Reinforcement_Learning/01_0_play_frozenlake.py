import gym
from gym.envs.registration import register

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _GetchWindows()

#Macro
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# key mapping
arrow_keys ={
    b'w' : UP,
    b's' : DOWN,
    b'd' : RIGHT,
    b'a' : LEFT}

#register frozenlake with is_slippery false
register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
env.render()

while True:
    key = getch()

    if key not in arrow_keys.keys():
        print("Game aborted!", key)
        print(arrow_keys)
        break

    action = arrow_keys[key] #에이전트의 움직임
    state, reward, done, info = env.step(action) #움직임에 대한 결과값들
    env.render() # 화면 출력
    print("State : ", state, "Action:", action, "Reward:", reward, "Info:",info)

    if done: #도착하면 게임을 끝낸다.
        print("Finished with reward", reward)
        break


